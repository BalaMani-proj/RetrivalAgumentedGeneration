from astrapy import DataAPIClient

# Initialize the client
client = DataAPIClient("AstraCS:ClLXZRKiZZnttAZEdScuzlfp:d249e772a2103f6c047ecc92dd7d3ef1886d6039be6e7191a725245fef59ac15")
db = client.get_database_by_api_endpoint(
  "https://5696ff70-0c34-4204-bc31-23483df0a269-us-east-2.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")