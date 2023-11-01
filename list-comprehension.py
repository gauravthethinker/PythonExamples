fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 256, -8, -4, -2, 5, -9]

#Exercise 1 - rewrite the above example code using list comprehension syntax. 
# Make a variable named uppercased_fruits to hold the output of the list comprehension. 
# Output should be ['MANGO', 'KIWI', etc...]

uppercased_fruits = [fruit.upper() for fruit in fruits]
print(uppercased_fruits)

# Exercise 2 - create a variable named capitalized_fruits and use list comprehension syntax to produce output 
# like ['Mango', 'Kiwi', 'Strawberry', etc...]

capitalized_fruits = [fruit.capitalize() for fruit in fruits]
print(capitalized_fruits)

# Exercise 3 - Use a list comprehension to make a variable named fruits_with_more_than_two_vowels.
def count_vowels(fruit):
    return sum(1 for letter in fruit if letter.lower() in 'aeiou')

fruits_with_more_than_two_vowels = [fruit for fruit in fruits if count_vowels(fruit) > 2]
print(fruits_with_more_than_two_vowels)

# Exercise 4 - make a variable named fruits_with_only_two_vowels. The result should be ['mango', 'kiwi', 'strawberry']

fruits_with_only_two_vowels = [fruit for fruit in fruits if count_vowels(fruit) == 2]
print(fruits_with_only_two_vowels)

#Exercise 5 - Make a variable named even_numbers that holds only the even numbers 
even_numbers = [num for num in numbers if num % 2 == 0]
print(even_numbers)

#Exercise 6 - Make a variable named odd_numbers that holds only the odd numbers
odd_numbers = [num for num in numbers if num % 2 != 0]
print(odd_numbers)

#Excercise 7 - use a list comprehension w/ a conditional in order to produce a list of numbers with 2 or more numerals
numbers_with_two_or_more_numerals = [num for num in numbers if abs(num) >= 10]
print(numbers_with_two_or_more_numerals)

#Excercise 8 - Make a variable named odd_negative_numbers that contains only the numbers that are both odd and negative.
odd_negative_numbers = [num for num in numbers if num < 0 and num % 2 != 0]
print(odd_negative_numbers)

#Excercise 9 - Make a variable named numbers_plus_5. In it, return a list containing each number plus five.
numbers_plus_5 = [num + 5 for num in numbers]
print(numbers_plus_5)

#Excercise 10 - Make a variable named fruits_with_letter_a that contains a list of only the fruits that contain the letter "a"
fruits_with_letter_a = [fruit for fruit in fruits if 'a' in fruit]
print(fruits_with_letter_a)