using OPFGenerator, PowerModels, PGLib, Ipopt, JuMP, MathOptInterface, MathOptSymbolicAD
using ChainRulesCore, DiffOpt

case = "1354_pegase"
optimizer = DiffOpt.diff_optimizer(Ipopt.Optimizer)

data = pglib(case) |> PowerModels.make_basic_network |> OPFGenerator.OPFData

function solve_opf(opf)
    optimize!(opf.model; _differentiation_backend = MathOptSymbolicAD.DefaultBackend())
    return value.(model[:pg]), value.(model[:pf]), value.(model[:va])
end

function ChainRulesCore.rrule(::typeof(build_and_solve), gs, b)
    data = deepcopy(data)
    data.gs .= gs
    data.b .= b
    opf = build_opf(OPFGenerator.DCOPF, data, optimizer)

    pg, pf, va = solve_opf(opf)
    function pullback(Δpg, Δpf, Δva)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), pg, Δpg)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), pf, Δpf)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), va, Δva)

        DiffOpt.reverse_differentiate!(JuMP.backend(opf.model))

        δkcl_p = JuMP.coefficient.(MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:kcl_p]))
        δohm_pf = JuMP.constant.(MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:ohm_pf]))
    end
    
end