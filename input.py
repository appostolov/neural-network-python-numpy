def email_to_array(email: str, max_length: int = 50) -> list:
    email = email.lower()
    flat_features = []
    
    for i in range(max_length):
        is_letter = 0
        is_number = 0
        is_at = 0
        is_dot = 0
        is_quote = 0
        is_special = 0
        is_padding = 0
        
        if i >= len(email):
            is_padding = 1
        else:
            char = email[i]
            if char == '@':
                is_at = 1
            elif char == '.':
                is_dot = 1
            elif char in ['"', "'"]:  # Captures both single and double quotes
                is_quote = 1
            elif char.isalpha():
                is_letter = 1
            elif char.isdigit():
                is_number = 1
            elif char in ['_', '-', '+']:
                is_special = 1
            else:
                is_special = 1 
        
        # Append the 7 binary switches for this position
        flat_features.extend([is_letter, is_number, is_at, is_dot, is_quote, is_special, is_padding])
        
    return flat_features