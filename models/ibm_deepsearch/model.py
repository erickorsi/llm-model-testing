import deepsearch as ds
api = ds.CpsApi.from_env()
print([p.name for p in api.projects.list()])