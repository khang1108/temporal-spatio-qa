# ==============================================================================
# Makefile biên dịch bài báo LaTeX với cơ chế tự động dọn dẹp
# ==============================================================================

.PHONY: all build auto-clean clean clean-all

all: build

build:
	@chmod +x build.sh
	@./build.sh build

auto-clean:
	@chmod +x build.sh
	@./build.sh --auto-clean

clean:
	@chmod +x build.sh
	@./build.sh clean

clean-all:
	@chmod +x build.sh
	@./build.sh clean-all
