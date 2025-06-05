"""
Neo4j Setup Instructions for LLM Resource Comparison Project

This file provides instructions for setting up Neo4j without Docker.

OPTION 1: Neo4j Desktop (Recommended for local development)
=========================================================
1. Download Neo4j Desktop from: https://neo4j.com/download/
2. Install and start Neo4j Desktop
3. Create a new database with these settings:
   - Database Name: llm-resource-comparison
   - Password: password (or your preferred password)
   - Version: Latest stable (5.x)
4. Start the database
5. Note the connection details (usually bolt://localhost:7687)

OPTION 2: Neo4j Aura (Cloud - Free tier available)
===================================================
1. Go to: https://neo4j.com/cloud/aura/
2. Create a free account
3. Create a new database
4. Note the connection URI and credentials provided

OPTION 3: Local Installation
============================
1. Download Neo4j Community Edition from: https://neo4j.com/download-center/
2. Extract and run:
   - Windows: bin\neo4j.bat console
   - Linux/Mac: bin/neo4j console
3. Access Neo4j Browser at http://localhost:7474
4. Set initial password

CONNECTION CONFIGURATION
========================
After setting up Neo4j, update the connection details in your code:

Default connection (Neo4j Desktop):
- URI: bolt://localhost:7687
- Username: neo4j
- Password: password (or what you set)

The migration script will use these default values unless you specify different ones.

NEXT STEPS
==========
1. Set up Neo4j using one of the options above
2. Run: python migrate_to_neo4j.py
3. Run the analytics and generate visualizations

SECURITY NOTE
=============
For production use, always use strong passwords and secure connections (bolt+s:// or neo4j+s://).
"""
