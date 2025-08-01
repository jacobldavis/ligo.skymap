#
# Copyright (C) 2025 Jacob Davis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

# Constant values from bayestar_skymap.h
import jax.numpy as jnp

dVC_dVL_data = jnp.array([
	-9.03530686e-04,
	-1.41061209e-03,
	-2.20206980e-03,
	-3.43704228e-03,
	-5.36324313e-03,
	-8.36565716e-03,
	-1.30408990e-02,
	-2.03097345e-02,
	-3.15839496e-02,
	-4.90067810e-02,
	-7.57823621e-02,
	-1.16591783e-01,
	-1.78043360e-01,
	-2.69015437e-01,
	-4.00642342e-01,
	-5.85684187e-01,
	-8.37285007e-01,
	-1.16766399e+00,
	-1.58760243e+00,
	-2.10699998e+00,
	-2.73543785e+00,
	-3.48101474e+00,
	-4.34716901e+00,
	-5.32990764e+00,
	-6.41818355e+00,
	-7.59706127e+00,
	-8.85110626e+00,
	-1.01663921e+01,
	-1.15311902e+01,
	-1.29359713e+01,
	-1.43731775e+01,
	-1.58369728e+01
])
dVC_dVL_tmin = 0.000000000000000
dVC_dVL_tmax = 13.815510557964274
dVC_dVL_dt = 0.445661630902073
dVC_dVL_high_z_slope = -3.304059176506592
dVC_dVL_high_z_intercept = 29.810291594530973