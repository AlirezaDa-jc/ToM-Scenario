from core.event import Event
from core.world import WorldState

if __name__ == "__main__":
 
    world = WorldState()
 
    agents = ["Alice", "Bob", "Charlie"]
    initial_objects = {"coin": "Box_A"}
 
    world.initialize(agents=agents, initial_objects=initial_objects)
 
    print("\nINITIAL STATE")
    world.print_state()
 
    event1 = Event(
        actor="Bob",
        action="move",
        target_object="coin",
        from_location="Box_A",
        to_location="Box_B",
        visible_to=["Bob", "Charlie"]
    )
 
    print("\n==============================")
    print(f"EVENT: {event1.actor} moves {event1.target_object} "
          f"from {event1.from_location} to {event1.to_location}")
    print(f"VISIBLE TO: {event1.visible_to}")
    print("==============================")
 
    world.process_event(event1)
    world.print_state()
 
    print("\n########################################")
    print("QUERIES")
    print("########################################")
 
    print("\nREALITY:")
    print("  coin =", world.query_world("coin"))
 
    print("\nFIRST ORDER:")
    for agent in agents:
        print(f"  {agent} believes coin =", world.query_first_order(agent, "coin"))
 
    print("\nSECOND ORDER:")
    print("  Charlie thinks Bob believes coin =",
          world.query_second_order("Charlie", "Bob", "coin"))
    print("  Alice thinks Bob believes coin =",
          world.query_second_order("Alice", "Bob", "coin"))