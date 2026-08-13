#!/usr/bin/env python3
# python/_archivist_mcp_server.py
"""MCP server for tutorial 12: the course-catalog database over stdio.

Exposes the SQLite course database built by 12_the_academy_archivist.py
as two MCP tools, so the tutorial's agent can reach structured catalog
data through a tool that lives in a different process. The underscore
prefix keeps this file out of the notebook build; it is a server script,
not a tutorial rung.

This server is deliberately written against the mcp SDK directly, not
fairlib's MCPServer: it stands in for the third-party server a capstone
does not control (a lab's database service, a vendor API), which is why
the tutorial's agent must discover its tools rather than import them.

Usage:
    python _archivist_mcp_server.py /path/to/courses.db

Both tools run parameterized queries against a read-only connection; no
caller-supplied text is ever interpolated into SQL.
"""
import asyncio
import os
import sqlite3
import sys
import urllib.parse
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def create_archives_server(db_path: str) -> Server:
    """Create the MCP server over the course-catalog database.

    Args:
        db_path: Path to the SQLite file built by the tutorial's
            extraction step. Opened read-only for every query.

    Returns:
        Configured MCP Server instance.
    """
    server = Server("usafa-archives")

    def query(sql: str, params: tuple) -> list[sqlite3.Row]:
        # quote() keeps path characters like ? and # from being read as
        # URI query syntax, so mode=ro cannot be accidentally overridden.
        conn = sqlite3.connect(
            f"file:{urllib.parse.quote(db_path)}?mode=ro", uri=True
        )
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="course_search",
                description=(
                    "Search the USAFA course-catalog database extracted from "
                    "the curriculum archives. Filter by any combination of a "
                    "keyword in the course title, a department name, a "
                    "catalog year, or a course number. Returns matching rows "
                    "as 'department number title (catalog_year)'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Substring to match in the course title.",
                        },
                        "department": {
                            "type": "string",
                            "description": "Department name, for example 'Engl' or 'Biology'.",
                        },
                        "catalog_year": {
                            "type": "string",
                            "description": "Catalog year label, for example 'AY1959-60'.",
                        },
                        "course_number": {
                            "type": "string",
                            "description": "Three-digit course number, for example '495'.",
                        },
                    },
                },
            ),
            Tool(
                name="catalog_summary",
                description=(
                    "Summarize the course-catalog database: how many courses "
                    "each department has in each catalog year. Takes no "
                    "arguments."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        arguments = arguments or {}
        if name == "course_search":
            clauses = []
            params: list[str] = []
            if arguments.get("keyword"):
                clauses.append("title LIKE ?")
                params.append(f"%{arguments['keyword']}%")
            if arguments.get("department"):
                # Catalogs abbreviate department names (Engl, Beh Sci), so
                # match in both directions: the stored name inside the
                # caller's phrase, or the caller's phrase inside the name.
                clauses.append(
                    "(department LIKE '%' || ? || '%' "
                    "OR ? LIKE '%' || department || '%')"
                )
                params.extend(
                    [arguments["department"], arguments["department"]]
                )
            if arguments.get("catalog_year"):
                clauses.append("catalog_year = ?")
                params.append(arguments["catalog_year"])
            if arguments.get("course_number"):
                clauses.append("number = ?")
                params.append(arguments["course_number"])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = query(
                "SELECT DISTINCT department, number, title, catalog_year "
                f"FROM courses {where} "
                "ORDER BY catalog_year, department, number LIMIT 40",
                tuple(params),
            )
            if not rows:
                return [TextContent(type="text", text="No courses matched.")]
            lines = [
                f"{r['department']} {r['number']} {r['title']} ({r['catalog_year']})"
                for r in rows
            ]
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "catalog_summary":
            rows = query(
                "SELECT catalog_year, department, COUNT(*) AS n FROM courses "
                "GROUP BY catalog_year, department ORDER BY catalog_year, n DESC",
                (),
            )
            lines = [
                f"{r['catalog_year']}: {r['department']} ({r['n']} courses)"
                for r in rows
            ]
            return [TextContent(type="text", text="\n".join(lines))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python _archivist_mcp_server.py /path/to/courses.db", file=sys.stderr)
        sys.exit(1)
    db_arg = sys.argv[1]
    if not os.path.isfile(db_arg):
        print(f"Database file not found: {db_arg}", file=sys.stderr)
        sys.exit(1)
    server = create_archives_server(db_arg)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
