SHADER  = shaders/tap_vanity.metal
SWIFT   = src/main.swift
BUILD   = build
METALLIB= $(BUILD)/default.metallib
AIR     = $(BUILD)/tap_vanity.air
BIN     = $(BUILD)/tapvanity_metal

SDK     = macosx
METALFLAGS = -O3
SWIFTFLAGS = -O -framework Metal -framework QuartzCore -framework Foundation -framework Security -framework CryptoKit

.PHONY: all clean test verify-bin run

all: $(BIN) $(METALLIB)

$(BUILD):
	mkdir -p $(BUILD)

$(AIR): $(SHADER) | $(BUILD)
	xcrun -sdk $(SDK) metal $(METALFLAGS) -c $(SHADER) -o $(AIR)

$(METALLIB): $(AIR)
	xcrun -sdk $(SDK) metallib $(AIR) -o $(METALLIB)

$(BIN): $(SWIFT) | $(BUILD)
	swiftc $(SWIFTFLAGS) -o $(BIN) $(SWIFT)

verify-bin:
	cd verify && cargo build --release

test: all verify-bin
	bash scripts/test.sh

clean:
	rm -rf $(BUILD) verify/target
