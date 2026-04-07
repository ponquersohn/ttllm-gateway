.PHONY: release release-minor release-major

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
LATEST_TAG := $(shell git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
MAJOR := $(shell echo $(LATEST_TAG) | sed 's/^v//' | cut -d. -f1)
MINOR := $(shell echo $(LATEST_TAG) | sed 's/^v//' | cut -d. -f2)
PATCH := $(shell echo $(LATEST_TAG) | sed 's/^v//' | cut -d. -f3)

define check_branch
	@if [ "$(BRANCH)" != "main" ]; then \
		echo "ERROR: Releases can only be created from 'main' branch (current: $(BRANCH))"; \
		exit 1; \
	fi
endef

define show_commands
	@echo ""
	@echo "Run the following commands to release $(1):"
	@echo ""
	@echo "  git tag -a $(1) -m \"Release $(1)\""
	@echo "  git push origin $(1)"
	@echo "  gh release create $(1) --title \"$(1)\" --generate-notes"
	@echo ""
endef

release: ## Patch release (v0.0.X)
	$(call check_branch)
	$(eval NEXT := v$(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    $(NEXT)"
	$(call show_commands,$(NEXT))

release-minor: ## Minor release (v0.X.0)
	$(call check_branch)
	$(eval NEXT := v$(MAJOR).$(shell echo $$(($(MINOR)+1))).0)
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    $(NEXT)"
	$(call show_commands,$(NEXT))

release-major: ## Major release (vX.0.0)
	$(call check_branch)
	$(eval NEXT := v$(shell echo $$(($(MAJOR)+1))).0.0)
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    $(NEXT)"
	$(call show_commands,$(NEXT))
