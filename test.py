def func_data_handiling(amount_paise: int):
    amount_rupee = amount_paise/100
    
    return f'₹{amount_rupee:.2f}'

print(func_data_handiling(123456))