def check_var_exists(var_name):
    if var_name in globals() or var_name in locals():
        return True # var exists.
    else:
        return False