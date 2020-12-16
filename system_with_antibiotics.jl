"""
Consider a ampicillin-resistant (population size denoted by e1, strain denoted by 1)
and chloramphenicol-resistant (population size denoted by e2, strain denoted by 2)
E. coli strains in the human body with their respective immune responses of
magnitudes x1 and x2 and antibiotic concentrations of ampicillin (A) and
chloramphenicol (C).
Inspired by "Evolutionary Dynamics" by Nowak, 2006, Section 10.1, and by
"Oscillatory dynamics in a bacterial cross-protection mutualism" by Yurtsev,
Conwill and Gore, 2016, we assume a well-mixed population consisting of e1, e2
and the immune cells of x1 and x2 and model its dynamics by

d/dt e1 = e1 r1(C) (1-γ/r1(C) (e1+e2)) - p x1 e1
d/dt e2 = e2 r2(A) (1-γ/r2(A) (e1+e2)) - p x2 e2
d/dt x1 = c e1 - b x1
d/dt x2 = c e2 - b x2
d/dt A  = -Vmax(C) A/(Km+A) e1(0)
d/dt C  = -c2(A) C e2(t)

where r1(C) and r2(C) are the intrinsic rates of reproduction of strain 1 and 2,
yielding their carrying capacities K1(C) = r1(C)/γ and K2(A) = r2(A)/γ,
Vmax(C) = 20,000 μg/mL 1/K1(C) is the hydrolysis rate of ampicillin,
Km = 12 μg/mL is the Michaelis-Menten constant for ampicillin inactivation,
c2(A) = 15 1/h 1/K2(A) is the inactivation rate of chloramphenicol (mass action).
The intrinsic rates of reproduction are 24h-periodic (fresh antibiotics with
concentrations A(0)=A₀ and C(0)=C₀ and fresh nutrients and dilution of a factor
of 100 of the whole mixed population every 24h) and modelled by

r1(C) =
    0                                t < tlag
    (β1-δ)/(1+C/IC50_12)             t ≧ tlag
r2(A) =
    0                                t < tlag
    -d2 + (β2-δ+d2)/(1+A/IC50_21)    t ≧ tlag

where IC50_12 = 1.3 μg/mL and IC50_21 = 0.65 μg/mL are the inhibitory
concentrations for chloramphenicol-resistant cells in ampicillin and for
ampicillin-resistant cells in chloramphenicol and
d2 = 0.25 1/h is the death rate of chloramphenicol-resistant cells at high
ampicillin concentrations. During lag time tlag = 1h no population growth
happens due to adaption to the new diluted environment.

In this code, this system of ODEs is solved using the classical Runge-Kutta
numerical method (RK4).
"""

using Plots
using PyPlot

# time discretization
t_end = 30 * 24.  # hours               # TODO play with it
n_timesteps = 100 * 30 * 24
n_timepoints = n_timesteps + 1
Δt = t_end / n_timesteps
t_array = collect(range(0, t_end, step=Δt))

# volume of media in which cells are grown
vol = 200          # μL

# parameters describing E. coli strains and immune responses
# intrinsic rates of reproduction of (unperturbed) system without antibiotics
β1_minus_δ = 1.18  # per hour
β2_minus_δ = 1.21  # per hour
# carrying capacity of strain 1 of system without antibiotics as density
κ1 = 2.3 * 10^6    # cells/μL
# carrying capacity of strain 1 of system without antibiotics
K1 = κ1 * vol
# competition rate through K1 = (β1-δ)/γ of system without antibiotics
γ = β1_minus_δ/K1

# immune response stimulation rate
c = 1//80 #1//80   # per hour,       0 -- 7/80, default: 1/16
# immune response decay rate
b = 1//8  #1//8    # per hour,       0 -- 7/24, default: 1/8
# bacteria elimination rate (mass action)
p = 4//240 #4//240  # per hour*cell,  1/240 -- 1/30, default: 7/240

# lag time after dilution
tlag = 1  # hour
# inhibitory concentrations
IC50_12 = 1.3  # μg/mL, for chloramphenicol-resistant cells in ampicillin
IC50_21 = 0.65 # μg/mL, for ampicillin-resistant cells in chloramphenicol
# death rate of chloramphenicol-resistant cells at high ampicillin concentrations
d2 = 0.0025  # per hour
# Michaelis-Menten constant for ampicillin inactivation
Km = 12  # μg/mL

# intial values
e1₀ = 0.#1*10^3
e2₀ = 1.5*10^3
x1₀ = 0.
x2₀ = 0.
A₀  = 3.24  # μg/mL     0 -- 50 # 0, 2, 16, 64, (120,) 256
C₀  = 145  # μg/mL      0 -- 25 # 0, 2, 10, 50, 250, 1250, 6250, 31250, 156250

# solution arrays
e1_array = zeros(Float64, n_timepoints)
e2_array = zeros(Float64, n_timepoints)
x1_array = zeros(Float64, n_timepoints)
x2_array = zeros(Float64, n_timepoints)
A_array = zeros(Float64, n_timepoints)
C_array = zeros(Float64, n_timepoints)

# initialize arrays
e1_array[1] = e1₀
e2_array[1] = e2₀
x1_array[1] = x1₀
x2_array[1] = x2₀
A_array[1]  = A₀
C_array[1]  = C₀

# function for the right-hand side
function rhs(e1 :: Float64, e2 :: Float64, x1 :: Float64, x2 :: Float64,
	         A :: Float64, C :: Float64, t :: Float64)
	# compute intrinsic rates of reproduction
	r1 = 0.
	r2 = 0.
	# compute hydrolysis rate of ampicillin
	Vmax = 0.
	# compute inactivation rate of chloramphenicol
	c2 = 0.
	if t >= tlag
		r1 = β1_minus_δ / (1 + C/IC50_12)
		r2 = -d2 + (β2_minus_δ + d2) / (1 + A/IC50_21)
		Vmax = 20000 / (r1/γ)  # μg/(mL h)  
		c2 = 15 / (r2/γ)       # per hour 
	end

    ė1 = 0.  # TODO also = 0 for immune cells?
	ė2 = 0.
	if t >= tlag
		ė1 = e1 * r1 * (1 - γ/r1 * (e1+e2)) - p * x1 * e1
		ė2 = e2 * r2 * (1 - γ/r2 * (e1+e2)) - p * x2 * e2
	end

	return [ė1, ė2,
			c * e1 - b * x1,
			c * e2 - b * x2,
			-Vmax * A/(Km+A) * e1₀,
			-c2 * C * e2] :: Array{Float64, 1}
end

n_days = round(Int, t_end/24.)
for d in 1:n_days  # day loop
	ti = 0.  # time after dilution: 0 -- 24h
	n_timesteps_ti = round(Int, n_timesteps/n_days)

	for i in 1:n_timesteps_ti   # day hour loop
		j = (d-1) * n_timesteps_ti + i  # index counting from day 1 hour 1 on
		# compute the weights for RK4
		ω1 = rhs(e1_array[j], e2_array[j], x1_array[j], x2_array[j], A_array[j], C_array[j], ti)
		ω2 = rhs(e1_array[i] + 0.5 * Δt * ω1[1], e2_array[j] + 0.5 * Δt * ω1[2],
		         x1_array[j] + 0.5 * Δt * ω1[3], x2_array[j] + 0.5 * Δt * ω1[4],
				 A_array[j] + 0.5 * Δt * ω1[5], C_array[j] + 0.5 * Δt * ω1[6],
				 ti + Δt/2)
		ω3 = rhs(e1_array[j] + 0.5 * Δt * ω2[1], e2_array[j] + 0.5 * Δt * ω2[2],
		         x1_array[j] + 0.5 * Δt * ω2[3], x2_array[j] + 0.5 * Δt * ω2[4],
				 A_array[j] + 0.5 * Δt * ω2[5], C_array[j] + 0.5 * Δt * ω2[6],
				 ti + Δt/2)
		ω4 = rhs(e1_array[j] + Δt * ω3[1], e2_array[j] + Δt * ω3[2],
		         x1_array[j] + Δt * ω3[3], x2_array[j] + Δt * ω3[4],
				 A_array[j] + Δt * ω3[5], C_array[j] + Δt * ω3[6],
				 ti + Δt)
		# formula for RK4: (x_{j+1} - x_j) / Δt = 1/6 (ω1 + 2 ω2 + 2 ω3 + ω4)
		e1_array[j+1] = e1_array[j] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[1]
		e2_array[j+1] = e2_array[j] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[2]
		x1_array[j+1] = x1_array[j] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[3]
		x2_array[j+1] = x2_array[j] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[4]
		A_array[j+1] = A_array[j] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[5]
		C_array[j+1] = C_array[j] + Δt/6 * (ω1 + 2 * ω2 + 2 * ω3 + ω4)[6]
	    # there can't be more than 0 and less than 1 bacterium in any strain
		if e1_array[j+1] < 1.
			e1_array[j+1] = 0.
		end
		if e2_array[j+1] < 1.
			e2_array[j+1] = 0.
		end
		# dilution at the beginning of the next day
		if i == n_timesteps_ti
			A_array[j+1] = A₀
			C_array[j+1] = C₀
			e1_array[j+1] = e1_array[j+1]/100
			e2_array[j+1] = e2_array[j+1]/100
			x1_array[j+1] = x1_array[j+1]/100
			x2_array[j+1] = x2_array[j+1]/100
			e1₀ = e1_array[j+1]
		end
		ti += Δt
	end
end

# plot the solution

#Plots.plot(t_array./24, e1_array, size=(1300, 700) ,xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes (E. coli strians) / magnitudes (immune responses)", color="gold", lab="e₁ (ampicillin-resistant E. coli)")
#Plots.plot!(t_array./24, e2_array, size=(1300, 700),xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes (E. coli strians) / magnitudes (immune responses)", color="blue", lab="e₂ (chloramphenicol-resistant E. coli)")
#Plots.plot!(t_array./24, x1_array, size=(1300, 700), xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes (E. coli strians) / magnitudes (immune responses)", color="magenta", lab="x₁ (immune response to e₁)")
#pic = Plots.plot!(t_array./24, x2_array, size=(1300,700),xlabel="t [days]", ylabel="populations [cells] or response magnitudes", title="Population sizes of E. coli strians / magnitudes (immune responses)", color="purple", lab="x₂ (immune response to e₂)")
#png(pic, "/tmp/system_with_antibiotics.png")
#using PyPlot
fig = PyPlot.figure(figsize=(12,4.5))
PyPlot.plot(t_array./24, e1_array, color="gold", label="e₁ (ampicillin-resistant E. coli)")
pic = PyPlot.plot(t_array./24, e2_array, color="blue", label="e₂ (chloramphenicol-resistant E. coli)")
#PyPlot.plot(t_array./24, x1_array, color="magenta", label="x₁ (immune response to e₁)")
#PyPlot.plot(t_array./24, x2_array, color="purple", label="x₂ (immune response to e₂)")
#PyPlot.plot(t_array./24, A_array, color="black", label="A (ampicillin concentration)")
#PyPlot.plot(t_array./24, C_array, color="grey", label="C (chloramphenicol concentration)")
#PyPlot.plot(t_array[500 * 27:n_timepoints]./24, e1_array[500 * 27:n_timepoints] ./ (e1_array[500 * 27:n_timepoints] + e2_array[500 * 27:n_timepoints]), color="green", label="e₁/(e₁+e₂)")
#PyPlot.plot(t_array[500 * 27:n_timepoints]./24, e2_array[500 * 27:n_timepoints] ./ (e1_array[500 * 27:n_timepoints] + e2_array[500 * 27:n_timepoints]), color="grey", label="e₂/(e₁+e₂)")
#pic = PyPlot.plot(t_array[500 * 27:n_timepoints]./24, e1_array[500 * 27:n_timepoints] ./ e2_array[500 * 27:n_timepoints], color="black", label="e₁/e₂")

PyPlot.title("Populations of E. coli strains (e₁(0)=$e1₀, e₂(0)=$(round(Int,e2₀)), A(0)=$A₀, C(0)=$C₀, b=$b, c=$c, p=$p)")  #/ magnitudes of immune responses (c=$c 1/h, b=$b 1/h, p=$p 1/(h*cell)
PyPlot.xlabel("t [days]")
PyPlot.ylabel("populations [cells]") # or response magnitudes
PyPlot.legend()
PyPlot.savefig("/tmp/system_with_antibiotics_pyplot_A$(A₀)_C$(C₀).png")
PyPlot.close()
