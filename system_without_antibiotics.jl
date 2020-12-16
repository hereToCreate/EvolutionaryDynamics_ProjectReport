"""
Consider a ampicillin-resistant (population size denoted by e1, strain denoted by 1)
and chloramphenicol-resistant (population size denoted by e2, strain denoted by 2)
E. coli strains in the human body with their respective immune responses of
magnitudes x1 and x2.
Inspired by "Evolutionary Dynamics" by Nowak, Section 10.1, we assume a
well-mixed population consisting of e1, e2 and the immune cells of x1 and x2
and model its dynamics by

d/dt e1 = e1(β1-δ) (1-γ/(β1-δ)(e1+e2)) - p x1 e1
     = e1(β1-δ) (1-(e1+e2)/K1) - p x1 e1
d/dt e2 = e2(β2-δ) (1-γ/(β2-δ)(e1+e2)) - p x2 e2
     = e2(β2-δ) (1-(e1+e2)/K2) - p x2 e2
d/dt x1 = c e1 - b x1
d/dt x2 = c e2 - b x2

with the carrying capacities of e1 and e2 given by
K1 = (β1-δ)/γ    and    K1 = (β1-δ)/γ

In this code, this system of ODEs is solved using the classical Runge-Kutta
numerical method (RK4).
"""

using Plots
using PyPlot

# time discretization
t_end = 360.                 # TODO play with it
n_timesteps = 1000
n_timepoints = n_timesteps + 1
Δt = t_end / n_timesteps
t_array = collect(range(0, t_end, step=Δt))

# volume of media in which cells are grown
vol = 200          # μL

# parameters describing E. coli strains and immune responses
# intrinsic rates of reproduction
β1_minus_δ = 1.18  # per hour
β2_minus_δ = 1.21  # per hour
# carrying capacity of strain 1 as density
κ1 = 2.3 * 10^6    # cells/μL
# carrying capacity of strain 1
K1 = κ1 * vol
# competition rate through K1 = (β1-δ)/γ
γ = β1_minus_δ/K1
# carrying capacity of strain 2 as density through K1 = (β1-δ)/γ and K2 = (β2-δ)/γ
κ2 = κ1 * β2_minus_δ/β1_minus_δ
# carrying capacity of strain 2
K2 = κ2 * vol
# immune response stimulation rate
c = 1//80  # 0 -- 7/80, default: 1/16
# immune response decay rate
b = 1//8   # 0 -- 7/24, default: 1/8
# bacteria elimination rate (mass action)
p = 4//240  # 1/240 -- 1/30, default: 7/240

# intial values
e1₀ = 1*10^3
e2₀ = 1.5*10^3
x1₀ = 0.
x2₀ = 0.

# solution arrays
e1_array = zeros(Float64, n_timepoints)
e2_array = zeros(Float64, n_timepoints)
x1_array = zeros(Float64, n_timepoints)
x2_array = zeros(Float64, n_timepoints)

# initialize arrays
e1_array[1] = e1₀
e2_array[1] = e2₀
x1_array[1] = x1₀
x2_array[1] = x2₀


# function for the right-hand side
function rhs(e1 :: Float64, e2 :: Float64, x1 :: Float64, x2 :: Float64)
	return [e1 * β1_minus_δ * (1-(e1+e2)/K1) - p * x1 * e1,
	        e2 * β2_minus_δ * (1-(e1+e2)/K2) - p * x2 * e2,
			c * e1 - b * x1,
			c * e2 - b * x2] :: Array{Float64, 1}
end

for i in 1:n_timesteps
	# compute the weights for RK4
	ω1 = rhs(e1_array[i], e2_array[i], x1_array[i], x2_array[i])
	ω2 = rhs(e1_array[i] + 0.5 * Δt * ω1[1], e2_array[i] + 0.5 * Δt * ω1[2],
	         x1_array[i] + 0.5 * Δt * ω1[3], x2_array[i] + 0.5 * Δt * ω1[4])
	ω3 = rhs(e1_array[i] + 0.5 * Δt * ω2[1], e2_array[i] + 0.5 * Δt * ω2[2],
	         x1_array[i] + 0.5 * Δt * ω2[3], x2_array[i] + 0.5 * Δt * ω2[4])
	ω4 = rhs(e1_array[i] + Δt * ω3[1], e2_array[i] + Δt * ω3[2],
	         x1_array[i] + Δt * ω3[3], x2_array[i] + Δt * ω3[4])
	# formula for RK4: (x_{i+1} - x_i) / Δt = 1/6 (ω1 + 2 ω2 + 2 ω3 + ω4)
	e1_array[i+1] = e1_array[i] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[1]
	e2_array[i+1] = e2_array[i] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[2]
	x1_array[i+1] = x1_array[i] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[3]
	x2_array[i+1] = x2_array[i] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[4]
    # there can't be more than 0 and less than 1 bacterium in any strain
	if e1_array[i+1] < 1.
		e1_array[i+1] = 0.
	end
	if e2_array[i+1] < 1.
		e2_array[i+1] = 0.
	end
end

# plot the solution

Plots.plot(t_array./24, e1_array, size=(1300, 700) ,xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes (E. coli strians) / magnitudes (immune responses)", color="gold", lab="e₁ (ampicillin-resistant E. coli)")
Plots.plot!(t_array./24, e2_array, size=(1300, 700),xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes (E. coli strians) / magnitudes (immune responses)", color="blue", lab="e₂ (chloramphenicol-resistant E. coli)")
Plots.plot!(t_array./24, x1_array, size=(1300, 700), xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes (E. coli strians) / magnitudes (immune responses)", color="magenta", lab="x₁ (immune response to e₁)")
pic = Plots.plot!(t_array./24, x2_array, size=(1300,700),xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes of E. coli strians / magnitudes (immune responses)", color="purple", lab="x₂ (immune response to e₂)")
png(pic, "/tmp/system_without_antibiotics.png")
#using PyPlot
fig = PyPlot.figure(figsize=(15,4.5))
PyPlot.plot(t_array./24, e1_array, color="gold", label="e₁ (ampicillin-resistant E. coli)")
PyPlot.plot(t_array./24, e2_array, color="blue", label="e₂ (chloramphenicol-resistant E. coli)")
PyPlot.plot(t_array./24, x1_array, color="magenta", label="x₁ (immune response to e₁)")
pic = PyPlot.plot(t_array./24, x2_array, color="purple", label="x₂ (immune response to e₂)")
PyPlot.title("Population sizes of E. coli strians / magnitudes of immune responses (c=$c 1/h, b=$b 1/h, p=$p 1/(h*cell)), e₁(0)=$e1₀, e₂(0)=$(round(Int,e2₀))")
PyPlot.xlabel("t [days]")
PyPlot.ylabel("populations [cells] or response magnitudes")
PyPlot.legend()
PyPlot.savefig("/tmp/system_without_antibiotics_pyplot.png")
PyPlot.close()
