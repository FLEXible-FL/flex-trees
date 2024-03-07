"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
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
