.PHONY: release release-minor release-major

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
LATEST_TAG := $(shell git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
MAJOR := $(shell echo $(LATEST_TAG) | sed 's/^v//' | cut -d. -f1)
MINOR := $(shell echo $(LATEST_TAG) | sed 's/^v//' | cut -d. -f2)
PATCH := $(shell echo $(LATEST_TAG) | sed 's/^v//' | cut -d. -f3)

VERSION_FILE := src/ttllm/__init__.py

define check_branch
	@if [ "$(BRANCH)" != "main" ]; then \
		echo "ERROR: Releases can only be created from 'main' branch (current: $(BRANCH))"; \
		exit 1; \
	fi
endef

define bump_version
	@sed -i 's/__version__ = ".*"/__version__ = "$(1)"/' $(VERSION_FILE)
	@echo "Bumped $(VERSION_FILE) to $(1)"
endef

define show_commands
	@echo ""
	@echo "Run the following commands to release v$(1):"
	@echo ""
	@echo "  git add $(VERSION_FILE)"
	@echo "  git commit -m \"Bump version to $(1)\""
	@echo "  git tag -a v$(1) -m \"Release v$(1)\""
	@echo "  git push origin main --tags"
	@echo "  gh release create v$(1) --title \"v$(1)\" --generate-notes"
	@echo ""
endef

release: ## Patch release (v0.0.X)
	$(call check_branch)
	$(eval NEXT := $(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    v$(NEXT)"
	$(call bump_version,$(NEXT))
	$(call show_commands,$(NEXT))

release-minor: ## Minor release (v0.X.0)
	$(call check_branch)
	$(eval NEXT := $(MAJOR).$(shell echo $$(($(MINOR)+1))).0)
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    v$(NEXT)"
	$(call bump_version,$(NEXT))
	$(call show_commands,$(NEXT))

release-major: ## Major release (vX.0.0)
	$(call check_branch)
	$(eval NEXT := $(shell echo $$(($(MAJOR)+1))).0.0)
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    v$(NEXT)"
	$(call bump_version,$(NEXT))
	$(call show_commands,$(NEXT))
