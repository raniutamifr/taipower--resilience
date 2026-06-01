ENV["PYTHON"] = "C:\\reXplan-repo\\Project Taipower\\.venv\\Scripts\\python.exe"
using Pkg
Pkg.build("PyCall")
println("PyCall rebuilt successfully!")