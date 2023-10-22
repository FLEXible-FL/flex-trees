from flex.actors.role_manager import FlexRoleManager


def select_client_by_id_from_pool(actor_id, actor_role, other_actor_id):
    """Function to select one client by the ID from the clients pool.
    Also it ensures that the actor_role is a client, as we are interested
    in selecting only clients.

    Args:
    ----
        actor_id (str): Actor ID from the FlexPool
        actor_role (FlexRole): Role of the actor_id
        other_actor_id (str): Actor ID to check

    Returns:
    -------
        Boolean: True/False whether actor_id is equal or not
        to the other_actor_id
    """
    if not FlexRoleManager.is_client(actor_role):
        return False
    return actor_id == other_actor_id


def select_client_neq_id_from_pool(actor_id, actor_role, neq_actor_id):
    """Function that select the clients which id is not equal than
    the actor given. Also it ensures that the actor_role is a client,
    as we are interested in selecting only clients.

    Args:
    ----
        actor_id (str): Actor ID from the FlexPool
        actor_role (FlexRole): Role of the actor_id
        neq_actor_id (str): Actor ID to check.
    """
    if not FlexRoleManager.is_client(actor_role):
        return False
    return actor_id != neq_actor_id
