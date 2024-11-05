using OPFGenerator, PowerModels, PGLib, Ipopt, JuMP, MathOptInterface, MathOptSymbolicAD
using HiGHS
using ChainRulesCore, DiffOpt

case = "1354_pegase"
optimizer = () -> DiffOpt.diff_optimizer(HiGHS.Optimizer)

data = pglib(case) |> PowerModels.make_basic_network |> OPFGenerator.OPFData

function solve_opf(opf)
    optimize!(opf.model; _differentiation_backend = MathOptSymbolicAD.DefaultBackend())
    return value.(opf.model[:pg]), value.(opf.model[:pf]), value.(opf.model[:va])
end

function build_and_solve(data, gs, b)
    data = deepcopy(data)
    data.gs .= gs
    data.b .= b
    opf = OPFGenerator.build_opf(OPFGenerator.DCOPF, data, optimizer)

    return solve_opf(opf)
end

function ChainRulesCore.rrule(::typeof(build_and_solve), data, gs, b)
    data = deepcopy(data)
    num_bus = length(data.gs)
    data.gs .= gs
    data.b .= b
    opf = OPFGenerator.build_opf(OPFGenerator.DCOPF, data, optimizer)

    pg, pf, va = solve_opf(opf)
    function pullback(Δpg, Δpf, Δva)
        # Set sensitivities for primal variables
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:pg], Δpg)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:pf], Δpf)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:va], Δva)
    
        # Run reverse differentiation
        DiffOpt.reverse_differentiate!(JuMP.backend(opf.model))
    
        # Compute derivatives for constraint rhs sensitivities
        δkcl_p = JuMP.constant.(MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:kcl_p]))
        δohm_pf = [sum(JuMP.coefficient.(MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:ohm_pf]), opf.model[:va][i])) for i in 1:num_bus]
    
        # Return derivatives with respect to `gs` and `b`
        dgs = δkcl_p  # adjust as per indexing in your model
        db = δohm_pf  # adjust as per indexing in your model
        return (NoTangent(), dgs, db)
    end
     return (pg, pf, va), pullback
end

len_gs, len_b = length(data.gs), length(data.b)

(pg, pf, va), pb = ChainRulesCore.rrule(build_and_solve, data, data.gs, data.b)

len_pg, len_pf, len_va = length(pg), length(pf), length(va)

Δpg, Δpf, Δva = fill(0.0, len_pg), fill(0.0, len_pf), fill(0.0, len_va)
Δpg[1] = 0.5

_, dgs, db = pb(Δpg, Δpf, Δva)



# using ChainRulesTestUtils
# test_rrule(build_and_solve, data, fill(0.0, len_gs), fill(0.0, len_b); tol=1e-6)