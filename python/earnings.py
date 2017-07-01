# Parameters earnings function:

# WAGECONSTAN: Utility of being employed
# ALPHA: Performance pay premium
# ZETA: Earnings premium
# GAMMA: Learning by doing
# XI: Depreciation of human capital

earnings_constants = [
    # Group 1
    dict(  
        WAGECONSTANT= 2.52, 
        ALPHA= 0.28,
        ZETA= [0, -.12, -.36, -.21, -.28, -.36, -.16, -.46, -.62],
        GAMMA= [0.76, 0.25, -0.0], 
        XI= -0.07
    ),
        
    # Group 2
    dict(  
        WAGECONSTANT= 3.60, 
        ALPHA= 0.17,
        ZETA= [0, -.29, -.48, -.26, -.40, -.50, -.31, -.53, -.72],
        GAMMA= [0.66, 0.22, 0.0], 
        XI= -0.08
    ),
        
    # Group 3
    dict(  
        WAGECONSTANT= 5.07, 
        ALPHA= -0.50,
        ZETA= [0, -.28, -.51, -.28, -.45, -.55, -.31, -.68, -.75],
        GAMMA= [0.61, 0.15, 0.0], 
        XI= -0.02
    ),
    
    # ALL
    dict(  
        WAGECONSTANT= 3.29, 
        ALPHA= 0.143,
        ZETA= [0, -0.31, -0.57, -0.28, -0.41, -0.55, -0.35, -0.63, -0.80],
        GAMMA= [0.758, 0.191, 0], 
        XI= -0.107
    )
]