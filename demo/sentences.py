# Shared sample set for the 3D map slide. Groups: food, places, tech, weather, sports.
SENTENCES = [
    ("food", "Appam with vegetable stew for breakfast"),
    ("food", "Puttu and kadala curry"),
    ("food", "Malabar chicken biryani"),
    ("food", "A strong filter coffee in the morning"),
    ("food", "Karimeen pollichathu wrapped in banana leaf"),
    ("food", "Where can I get a good sadya on Onam?"),
    ("places", "Houseboat cruise on the Alleppey backwaters"),
    ("places", "Tea gardens of Munnar in the mist"),
    ("places", "Chinese fishing nets at Fort Kochi"),
    ("places", "Sunset at Varkala cliff"),
    ("places", "Wildlife safari in Thekkady"),
    ("places", "Weekend trip from Kochi to the hills"),
    ("tech", "Restart the crashed Kubernetes pod"),
    ("tech", "Python script throws a KeyError"),
    ("tech", "Deploying a container to Cloud Run"),
    ("tech", "The GPU ran out of memory during training"),
    ("tech", "Write a Go function to parse JSON"),
    ("tech", "Our API latency spiked after the release"),
    ("weather", "Heavy monsoon rain expected in Ernakulam"),
    ("weather", "Orange alert issued for coastal districts"),
    ("weather", "മഴ ശക്തമായി പെയ്യുന്നു"),
    ("weather", "Carry an umbrella, it is pouring"),
    ("weather", "Humid and cloudy afternoon"),
    ("weather", "Waterlogging on MG Road after the downpour"),
    ("sports", "India won the test match by an innings"),
    ("sports", "Kerala Blasters scored in the final minute"),
    ("sports", "Sanju Samson hits a century"),
    ("sports", "Snake boat race at Punnamada lake"),
    ("sports", "Football fans packed the Kaloor stadium"),
    ("sports", "Practising cover drives at the nets"),
]

# Queries dropped onto the map during the story. First item is the topic they should land near
# (used only for the illustrative layout; real mode places them by their actual embeddings).
QUERIES = [
    ("food", "Where should I eat breakfast?"),
    ("weather", "കൊച്ചിയിൽ ഇന്ന് മഴ പെയ്യുമോ?"),  # "Will it rain in Kochi today?"
    ("tech", "My container keeps crashing"),
]
