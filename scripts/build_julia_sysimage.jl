"""
Build a Julia sysimage with SymbolicRegression.jl precompiled.

This eliminates the ~4-minute Julia precompilation on first PySR import.
Run via the shell wrapper: ./scripts/build_julia_sysimage.sh

Requires Julia >= 1.9 and PackageCompiler.jl.
"""

using Pkg

# Ensure PackageCompiler is installed
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    Pkg.add("PackageCompiler")
end

# Ensure SymbolicRegression is installed (PySR's backend)
if !haskey(Pkg.project().dependencies, "SymbolicRegression")
    Pkg.add("SymbolicRegression")
end

using PackageCompiler
using SymbolicRegression

# Precompile statements: exercise the code paths PySR uses
precompile_script = tempname() * ".jl"
open(precompile_script, "w") do f
    write(f, """
    using SymbolicRegression

    # Minimal SR run to exercise code paths
    X = randn(Float32, 5, 100)
    y = X[1, :] .* 2.0f0 .+ X[2, :] .^ 2

    options = SymbolicRegression.Options(;
        binary_operators=[+, -, *, /],
        unary_operators=[sin, cos],
        populations=3,
        population_size=20,
        maxsize=10,
        timeout_in_seconds=30,
    )

    hall_of_fame = equation_search(
        X, y;
        options=options,
        niterations=2,
        parallelism=:serial,
    )
    """)
end

output_path = joinpath(@__DIR__, "..", ".julia_sysimage.so")

println("Building sysimage at: ", output_path)
println("This will take several minutes...")

create_sysimage(
    [:SymbolicRegression];
    sysimage_path=output_path,
    precompile_statements_file=precompile_script,
)

rm(precompile_script; force=true)

println("\nSysimage built successfully: ", output_path)
println("Set JULIA_SYSIMAGE_PATH=", output_path, " before running PySR.")
