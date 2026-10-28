"""Learning Jev.

**Jev** is TypeSafe's first *System One* model: ``state`` plus typed questions
in, typed decisions out. Six lessons in ``jev_tutorial.ipynb``, plus:

* ``jev_client.py`` — the gateway client: typed questions in, typed answers out
* ``router.py``     — the smallest useful Jev app, a support-ticket router

Jev is reached through the Databricks Unity AI Gateway, so there is nothing
extra to install — only three values in ``jevai/.env``::

    DATABRICKS_HOST=...
    DATABRICKS_TOKEN=...
    JEV_MODEL=catalog.schema.jev_endpoint    # the Unity Catalog endpoint path
"""
