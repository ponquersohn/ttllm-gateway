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

define do_release
	@echo "Current version: $(LATEST_TAG)"
	@echo "Next version:    v$(1)"
	@echo ""
	@read -p "Create release v$(1)? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@git tag -a "v$(1)" -m "Release v$(1)"
	@echo ""
	@echo "Tag v$(1) created. Run the following to publish:"
	@echo ""
	@echo "  git push origin main --tags"
	@echo "  gh release create v$(1) --title \"v$(1)\" --generate-notes"
	@echo ""
endef

release: ## Patch release (v0.0.X)
	$(call check_branch)
	$(eval NEXT := $(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))
	$(call do_release,$(NEXT))

release-minor: ## Minor release (v0.X.0)
	$(call check_branch)
	$(eval NEXT := $(MAJOR).$(shell echo $$(($(MINOR)+1))).0)
	$(call do_release,$(NEXT))

release-major: ## Major release (vX.0.0)
	$(call check_branch)
	$(eval NEXT := $(shell echo $$(($(MAJOR)+1))).0.0)
	$(call do_release,$(NEXT))
