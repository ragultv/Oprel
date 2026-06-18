from oprel.server import db
from oprel.core.groups.lanes import LaneResult

def sequence_and_commit_results(group_id: str, round_id: str, lane_results: list[LaneResult]) -> list[dict]:
    """
    Takes an unordered list of results from the concurrent execution lanes,
    sorts them deterministically based on agent priority_order, and commits
    them to the database with monotonically increasing sequence numbers.
    """
    if not lane_results:
        return []
        
    # Sort results deterministically by priority_order
    # Lower number = higher priority = earlier in the sequence
    sorted_results = sorted(lane_results, key=lambda r: r.member.get("priority_order", 0))
    
    # Get current max sequence number to append strictly after existing messages in the round
    current_max_seq = db.get_max_sequence_number(group_id, round_id)
    
    committed_messages = []
    
    for idx, result in enumerate(sorted_results):
        seq_num = current_max_seq + 1 + idx
        msg_type = "interrupt" if result.decision.decision == "interrupt" else "reply"
        
        # Commit to database
        msg = db.add_group_message(
            group_id=group_id,
            round_id=round_id,
            sender_type="agent",
            member_id=result.member["id"],
            content=result.text.strip(),
            message_type=msg_type,
            sequence_number=seq_num
        )
        committed_messages.append(msg)
        
    return committed_messages
