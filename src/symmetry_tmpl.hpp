
/*
 * pyblock3: An Efficient python MPS/DMRG Library
 * Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#ifdef TMPL_NAME

#define STRINGIFY(M) #M
#define STRINGIFY2(M) STRINGIFY(M)
#define CAT(X, Y) X##Y
#define NAME_IMPL(X, Y) STRINGIFY2(CAT(X,Y))

// explicit template instantiation for each symmetry

#include "sz.hpp"
#define TMPL_Q SZ
#include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
#undef TMPL_Q

// add other symmetries here ...

#include "fermion_symmetry.hpp"
#define TMPL_Q U11
#include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
#undef TMPL_Q

#define TMPL_Q U1
#include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
#undef TMPL_Q

#define TMPL_Q Z2
#include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
#undef TMPL_Q

#define TMPL_Q Z4
#include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
#undef TMPL_Q

#define TMPL_Q Z22
#include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
#undef TMPL_Q


#undef NAME_IMPL
#undef CAT
#undef STRINGIFY2
#undef STRINGIFY

#endif
