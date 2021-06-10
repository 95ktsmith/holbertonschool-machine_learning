#!/usr/bin/env python3
""" Question and Answer loop """

if __name__ == "__main__":
    while True:
        question = input("Q: ").lower()
        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        print("A: ")
