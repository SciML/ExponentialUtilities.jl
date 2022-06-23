using Documenter, ExponentialUtilities

include("pages.jl")

makedocs(sitename = "ExponentialUtilities.jl",
         authors = "Chris Rackauckas",
         modules = [ExponentialUtilities],
         clean = true, doctest = false,
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://exponentialutilities.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/ExponentialUtilities.jl.git";
           push_preview = true)
