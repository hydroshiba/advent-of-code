# Advent of Code

## What is this about?

From the [Advent of Code](https://adventofcode.com/about) website:

> Advent of Code is an Advent calendar of small programming puzzles for a variety of skill levels that can be solved in any programming language you like. People use them as interview prep, company training, university coursework, practice problems, a speed contest, or to challenge each other.

The fun thing about these challenges are, there is no *right* way on how to solve these! This means AoC are more than just puzzles; they are also opportunities for the participants to try out new skills (and using them to solve them).

I also have a fair share amount of skills I want to try out, and AoC is the perfect mean for me to do so. Below are my solutions to AoC puzzles, *in various ways*!

## Running the solutions

I've make a simple Makefile to build and run the solutions. The command to run a solution is as follows:

```bash
make {day}-{part}-{lang}
```

This will default to using `STDIN` for the input. Since the input of these puzzles usually doesn't have the amount of lines specified, the only way to know if the input ends is to detect `EOF`, so it is strongly recommended to use an input file instead of typing manually from `STDIN`, either by using pipe or just simply input redirection:

```bash
# pipe
cat input.txt | make 1-1-cpp

# input redirection
make 1-1-cpp < input.txt
```

## Table of solutions

| Puzzle | Solution           | Part 1            | Part 2            |
|--------|--------------------|-------------------|-------------------|
| Day 1  | [Day 1](01/sol.md) | [cpp](01/1-1.cpp) | [cpp](01/1-2.cpp) |
| Day 2  |                    |                   |                   |
| Day 3  |                    |                   |                   |
| Day 4  |                    |                   |                   |
| Day 5  |                    |                   |                   |
| Day 6  |                    |                   |                   |
| Day 7  |                    |                   |                   |
| Day 8  |                    |                   |                   |
| Day 9  |                    |                   |                   |
| Day 10 |                    |                   |                   |
| Day 11 |                    |                   |                   |
| Day 12 |                    |                   |                   |
