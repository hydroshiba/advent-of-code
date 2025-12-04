# Terminal text formatting

.PHONY: quiet_message default
RED     := \033[31m
GREEN   := \033[32m
YELLOW  := \033[33m
BLUE    := \033[34m
MAGENTA := \033[35m
CYAN    := \033[36m
WHITE   := \033[37m
BOLD    := \033[1m
RESET   := \033[0m

# Language compilers and flags

CXX = $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null)
CXXFLAGS = -O2 -std=c++17

CUDA = nvcc
CUDAFLAGS = -O2 -std=c++17

PAS = $(shell which fpc 2>/dev/null || which gpc 2>/dev/null || which ppc 2>/dev/null || which ppca64 2>/dev/null)
PASFLAGS = -O2 -Mdelphi -v0 -k-w -FU.tmp -FE. -l-

PY = python3
CS = dotnet

# Directories

SOURCEDIRS = $(shell find . -type d)
VPATH = $(SOURCEDIRS)
INCLUDE = -I./helper

# Makefile rules

default:
	@printf "\n$(BOLD)$(GREEN)Advent $(RED)of $(GREEN)Code $(WHITE)Makefile utility$(RESET)\n"
	@printf "	$(BOLD)Usage: $(RESET)$(MAGENTA)make $(WHITE)$(BOLD){day}$(RESET)-$(WHITE)$(BOLD){part}$(RESET)-$(WHITE)$(BOLD){lang}\n"
	@printf "	$(BOLD)Example: $(RESET)$(MAGENTA)make $(RESET)1-2-cpp\n\n"

%: default

%-cpp: %.cpp
	@$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@.out
	@./$@.out

%-cu: %.cu
	@$(CUDA) $(CUDAFLAGS) $(INCLUDE) $< -o $@.out
	@./$@.out

%-py: %.py
	@$(PY) $<

%-cs: %.cs
	@if [ ! -d ".cs_runner" ]; then $(CS) new console -o .cs_runner > /dev/null; fi
	@cp $< .cs_runner/Program.cs
	@$(CS) run --project .cs_runner --no-restore

%-pas: %.pas
	@if [ ! -d ".tmp" ]; then mkdir .tmp; fi
	@$(PAS) $(PASFLAGS) $< -o$@.out
	@rm -rf .tmp
	@./$@.out

.DEFAULT:
	@$(MAKE) --no-print-directory default
	@printf "$(RED)Error: The target '$@' is not valid.$(RESET)\n"