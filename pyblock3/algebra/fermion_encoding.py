from .fermion_symmetry import U11, U1, Z2, Z4, Z22
from .fermion_setting import symmetry_map

state_map_u11 = {0:(U11(0),0,1),
                1:(U11(1,1),0,1),
                2:(U11(1,-1),0,1),
                3:(U11(2,0),0,1)}

state_map_u1 = {0:(U1(0),0,1),
                1:(U1(1),0,2),
                2:(U1(1),1,2),
                3:(U1(2),0,1)}

state_map_z4 = {0:(Z4(0),0,2),
                1:(Z4(1),0,2),
                2:(Z4(1),1,2),
                3:(Z4(0),1,2)}

state_map_z2 = {0:(Z2(0),0,2),
                1:(Z2(1),0,2),
                2:(Z2(1),1,2),
                3:(Z2(0),1,2)}

state_map_z22 = {0:(Z22(0),0,1),
                1:(Z22(1,0),0,1),
                2:(Z22(1,1),0,1),
                3:(Z22(0,1),0,1)}

cre_map = {0:"", 1:"+", 2:"-", 3:"+-"}
ann_map = {0:"", 1:"+", 2:"-", 3:"-+"}
hop_map = {0:(1,2), 1:(0,3), 2:(0,3), 3:(1,2)}

sz_dict = {0:0, 1:.5, 2:-.5, 3:0}
pn_dict = {0:0, 1:1,  2:1,   3:2}

def get_state_map(symmetry):
    if isinstance(symmetry, str):
        symmetry_string = symmetry.upper()
    else:
        symmetry_string = symmetry.__name__
    return {"U11": state_map_u11,
            "U1": state_map_u1,
            "Z4": state_map_z4,
            "Z22": state_map_z22,
            "Z2": state_map_z2}[symmetry_string]
