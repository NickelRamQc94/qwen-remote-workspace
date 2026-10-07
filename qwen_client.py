#!/usr/bin/env python3
"""Qwen integration client for the centralized remote workspace hub."""

import argparse
import json
import os
import sys
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_MARKER = "❤️94"
DEFAULT_GITHUB_USER = "NickelRamQc94"


def _read_repo_list() -> List[str]:
    """Return the configured target repositories as a list of repository identifiers."""
    raw = os.getenv("REPOSITORIES") or os.getenv("TARGET_REPOSITORIES") or ""
    repos = []
    for item in raw.split(","):
        cleaned = item.strip()
        if cleaned:
            repos.append(cleaned)
    return repos


def _normalize_repo_names(repos: Optional[List[str]] = None) -> List[str]:
    """Normalize repo names to full names: user/repo or repo if already qualified."""
    target_repos = repos if repos is not None else _read_repo_list()
    github_user = (os.getenv("GITHUB_USER") or DEFAULT_GITHUB_USER).strip()
    normalized: List[str] = []

    for repo in target_repos:
        repo_name = repo.strip()
        if not repo_name:
            continue
        if "/" in repo_name:
            normalized.append(repo_name)
        else:
            normalized.append(f"{github_user}/{repo_name}")

    return normalized


class QwenClient:
    """Small OpenAI-compatible client for the Alibaba Qwen API."""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (api_key or os.getenv("QWEN_API_KEY") or "").strip()
        self.api_url = (api_url or os.getenv("QWEN_API_URL") or DEFAULT_QWEN_API_URL).rstrip("/")
        self.model = (model or os.getenv("QWEN_MODEL") or DEFAULT_QWEN_MODEL).strip()
        self.marker = (os.getenv("MARKER") or DEFAULT_MARKER).strip()

        if not self.api_key:
            raise ValueError("QWEN_API_KEY is missing. Set it in the environment or .env file.")

    def _chat_completion_url(self) -> str:
        return f"{self.api_url}/chat/completions"

    def _build_system_prompt(self) -> str:
        return (
            "Tu es un assistant DevOps / GitHub qui agit selon le Protocole TABARNAK. "
            "Ta priorité est la sécurité, la clarté, la fiabilité et la conformité aux bonnes pratiques. "
            "Traite les fichiers, les flux CI/CD, les secrets et les actions GitHub comme des éléments sensibles. "
            "Si une action implique un secret, dis-le explicitement sans jamais l'exposer. "
            "Tu dois répondre en français, de façon concise et utile, avec une structure claire. "
            f"Ajoute toujours le marqueur {self.marker} à la fin de chaque réponse."
        )

    def ask(self, prompt: str) -> str:
        """Send a prompt to Qwen and return the textual answer."""
        if not prompt or not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": prompt.strip()},
            ],
            "temperature": 0.2,
            "max_tokens": 2000,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self._chat_completion_url(),
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )

        if response.status_code >= 400:
            try:
                error_body = response.json()
            except ValueError:
                error_body = {"error": response.text}
            message = error_body.get("error", {}).get("message") if isinstance(error_body.get("error"), dict) else error_body
            raise RuntimeError(f"Qwen API error ({response.status_code}): {message}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid JSON returned by the Qwen API.") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by the Qwen API.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("The Qwen API response did not include any text content.")

        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        return str(content).strip()

    def list_target_repositories(self) -> List[str]:
        """Return the list of target repositories as full GitHub names."""
        return _normalize_repo_names(_read_repo_list())


def main() -> int:
    parser = argparse.ArgumentParser(description="Client Qwen pour le hub multi-dépôts GitHub.")
    parser.add_argument("--prompt", help="Prompt à envoyer à Qwen.")
    parser.add_argument("--list-repos", action="store_true", help="Liste les dépôts configurés.")
    args = parser.parse_args()

    try:
        client = QwenClient()

        if args.list_repos:
            print(json.dumps(client.list_target_repositories(), indent=2))
            return 0

        if not args.prompt:
            prompt = sys.stdin.read().strip()
        else:
            prompt = args.prompt.strip()

        if not prompt:
            raise ValueError("Aucun prompt n'a été fourni. Utilisez --prompt ou stdin.")

        response = client.ask(prompt)
        print(response)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
