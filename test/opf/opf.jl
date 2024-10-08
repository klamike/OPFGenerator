using OPFGenerator: OPFData

include("utils.jl")
include("ptdf.jl")

function test_opf_pm(OPF::Type{<:OPFGenerator.AbstractFormulation}, casename::String)
    network = make_basic_network(pglib(casename))
    test_opf_pm(OPF, network)
end

"""
    test_opf_pm(OPF, data)

Build & solve opf using OPFGenerator, and compare against PowerModels implementation.
"""
function test_opf_pm(::Type{OPF}, data::Dict) where{OPF <: OPFGenerator.AbstractFormulation}
    error("""`test_opf_pm($(OPF), data)` not implemented.
    You must implement a function with the following signature:
        function test_opf_pm(::Type{OPF}, data::Dict) where{OPF <: $(OPF)}
            # unit tests ...
            return nothing
        end
    """)

    return nothing
end

include("acp.jl")
include("dcp.jl")
include("socwr.jl")
include("ed.jl")

# other tests
include("quad_obj.jl")

const PGLIB_CASES = ["14_ieee", "30_ieee", "57_ieee", "89_pegase", "118_ieee"]

@testset "OPF" begin
    @testset "$(OPF)" for OPF in OPFGenerator.SUPPORTED_OPF_MODELS
        @testset "$(casename)" for casename in PGLIB_CASES
            network = make_basic_network(pglib(casename))
            test_opf_pm(OPF, network)
        end

        @testset "QuadObj" begin test_quad_obj_warn(OPF) end
    end


    @testset _test_socwr_DualSolFormat()

    @testset _test_socwr_DualFeasibility()
end


@testset "OPFData" begin
    test_voltage_phasor_bounds_scalar()
    @testset "$casename" for casename in PGLIB_CASES
        network = make_basic_network(pglib(casename))
        data = OPFData(network)
        test_opfdata(data, network)
        test_voltage_phasor_bounds(data, network)
        @testset "PTDF" begin
            test_ptdf(network, data)
        end
    end
end
