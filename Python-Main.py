MAX_LINES = 3
MAX_BET = 100
MIN_BET = 1

def deposit():
    while True:
        amount = input("What would you like to deposit ? $")
        if amount.isdigit():
            amount = int(amount)
            if amount > 0:
                break
            else:
                print("Amount must be greater than 0")
        else:
            print("Please enter a number")

    print("You have entered valid amount")
    return amount

def get_number_of_line():
    while True:
        lines = input("Enter the number of line to bet on (1-"+str(MAX_LINES)+ ")? ")
        if lines.isdigit():
            lines = int(lines)
            if 1 <= lines <= MAX_LINES:
                break
            else:
                print("Enter a valid number of Lines")
        else:
            print("Please enter a number")

    return lines

def get_bet():
    while True:
        amount = input("What would you like to bet ? $")
        if amount.isdigit():
            amount = int(amount)
            if MIN_BET <= amount <= MAX_BET:
                break
            else:
                print(f"Amount must be betweem ${MIN_BET} - ${MAX_BET}.")
        else:
            print("Please enter a number")

    return amount


def main():
    balance = deposit()
    lines = get_number_of_line()
    bet = get_bet()
    total_bet = bet * lines
    print(f"You are betting to ${bet} on {lines}. Total bet is equal to ${total_bet}")
    
main()