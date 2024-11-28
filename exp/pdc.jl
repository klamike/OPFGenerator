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
    function pullback(ΔΩ)
        Δpg, Δpf, Δva = ΔΩ
        # Set sensitivities for primal variables
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:pg], Δpg)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:pf], Δpf)
        MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:va], Δva)
    
        # Run reverse differentiation
        DiffOpt.reverse_differentiate!(JuMP.backend(opf.model))
    

        # Compute derivatives for constraint rhs sensitivities
        δkcl_p = JuMP.constant.(MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:kcl_p]))
        brs = MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:ohm_pf])
        δohm_pf = zeros(length(brs))
        for i in 1:length(brs)
            for j in 1:num_bus
                val = JuMP.coefficient.(brs[i], opf.model[:va][j])
                if val != 0
                    δohm_pf[i] = - abs(val)
                    break
                end
            end
        end

        # Return derivatives with respect to `gs` and `b`
        dgs = δkcl_p  # adjust as per indexing in your model
        db = δohm_pf  # adjust as per indexing in your model
        return (NoTangent(), dgs, db)
    end
     return (pg, pf, va), pullback
end


include("exp/nlp_utilities.jl")

function build_and_solve_NLP(data, gs, b)
    data = deepcopy(data)
    data.gs .= gs
    data.b .= b
    opf = OPFGenerator.build_opf(OPFGenerator.DCOPF, data, Ipopt.Optimizer)

    return solve_opf(opf)
end

function ChainRulesCore.rrule(::typeof(build_and_solve_NLP), data, _gs, _b)
    optimizer = Ipopt.Optimizer
    model = JuMP.GenericModel{Float64}(optimizer)
    data = deepcopy(data)
    num_bus = length(data.gs)
    @variable(model, gs[i=1:length(_gs)] ∈ MOI.Parameter.(_gs[i]))
    @variable(model, b[i=1:length(_b)] ∈ MOI.Parameter.(_b[i]))
    data.gs = gs
    data.b = b
    opf = OPFGenerator.build_opf(OPFGenerator.DCOPF, data, optimizer; model=model)

    pg, pf, va = solve_opf(opf)
    focus_vars = [opf.model[:pg]; opf.model[:pf]; opf.model[:va]]
    primal_vars = all_primal_vars(opf.model)
    vars_idx = [findall(x -> x == i, primal_vars)[1] for i in focus_vars]
    params= [gs; b]
    evaluator, cons = create_evaluator(opf.model; x=[primal_vars; params])

    function pullback(ΔΩ)
        Δpg, Δpf, Δva = ΔΩ
        Δp= nothing
        # Set sensitivities for primal variables
        # MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:pg], Δpg)
        # MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:pf], Δpf)
        # MOI.set.(opf.model, DiffOpt.ReverseVariablePrimal(), opf.model[:va], Δva)
    
        # Run reverse differentiation
        # DiffOpt.reverse_differentiate!(JuMP.backend(opf.model))
        Δs, sp = compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=params)
        dΩ_dbgs = Δs[vars_idx, :]'vcat(Δpg, Δpf, Δva)
        # Compute derivatives for constraint rhs sensitivities
        # δkcl_p = JuMP.constant.(MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:kcl_p]))
        # brs = MOI.get.(opf.model, DiffOpt.ReverseConstraintFunction(), opf.model[:ohm_pf])
        # δohm_pf = zeros(length(brs))
        # for i in 1:length(brs)
        #     for j in 1:num_bus
        #         val = JuMP.coefficient.(brs[i], opf.model[:va][j])
        #         if val != 0
        #             δohm_pf[i] = - abs(val)
        #             break
        #         end
        #     end
        # end
    
        # Return derivatives with respect to `gs` and `b`
        dgs = dΩ_dbgs[1:length(_gs)]
        db = dΩ_dbgs[length(_gs)+1:end]
        return (NoTangent(), dgs, db)
    end
     return (pg, pf, va), pullback
end

# len_gs, len_b = length(data.gs), length(data.b)

# (pg, pf, va), pb = ChainRulesCore.rrule(build_and_solve_NLP, data, data.gs, data.b)

# ### Test the obvious
# pg_p, pf_p, va_p = build_and_solve_NLP(data, data.gs, data.b)

# pg ≈ pg_p
# pf ≈ pf_p
# va ≈ va_p
# ###

# # ### Test the reverse mode
# len_pg, len_pf, len_va = length(pg), length(pf), length(va)

# Δpg, Δpf, Δva = fill(1.0, len_pg), fill(1.0, len_pf), fill(1.0, len_va)
# # Δpg[1] = 1


# _, dgs, db = pb((Δpg, Δpf, Δva))

# dgx_idx, idx = findmax(abs.(dgs))

# deltal_gs = zeros(len_gs)
# deltal_gs[idx] = 0.001

# pg_p, pf_p, va_p = build_and_solve_NLP(data, data.gs + deltal_gs, data.b)

# pg_p[1] ≈ pg[1] + dgx_idx * 0.001 # fails but is very close (expected)



using ChainRulesTestUtils
test_rrule(build_and_solve, data ⊢ NoTangent(), data.gs, data.b; check_inferred=false)
