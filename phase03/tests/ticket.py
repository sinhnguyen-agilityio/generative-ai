from middlewares.security_pipeline import SecurityPipeline

ticket = "My email is john@gmail.com. You idiots!"

security_pipeline = SecurityPipeline().build()
secure_ticket = security_pipeline.invoke(ticket)

print(secure_ticket.model_dump_json(indent=2))
