using Documenter
using FactorVSA

DocMeta.setdocmeta!(FactorVSA, :DocTestSetup, :(using FactorVSA); recursive=true)

makedocs(;
    modules=[FactorVSA],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("CognitiveSubstratesAI", "FactorVSA"),
    sitename="FactorVSA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cognitivesubstratesai.github.io/FactorVSA/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=["Home" => "index.md"],
    # index.md links to SPEC.md / GATE_RESULT.md outside docs/src; tolerate warnings.
    warnonly=true
)

deploydocs(; repo="github.com/CognitiveSubstratesAI/FactorVSA", devbranch="main")
