setup-local-env:
	@echo ""
	@echo "🚇 Starting port-forwards..."
	@kubectl port-forward svc/my-lib-bot-qdrant 6333:6333 > /dev/null 2>&1 & \
	kubectl port-forward svc/my-lib-bot-qdrant 6334:6334 > /dev/null 2>&1 & \
	kubectl port-forward svc/my-lib-bot-qdrant 6335:6335 > /dev/null 2>&1 & \
	kubectl port-forward svc/my-lib-bot-mongodb 27017:27017 > /dev/null 2>&1 & \
	echo "🧠 Qdrant available at http://localhost:6333" && \
	echo "🍃 MongoDB available at mongodb://localhost:27017" && \
	echo "✅ Port-forwards started."
	@echo ""