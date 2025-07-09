#!/bin/bash
DEST=".."

cp -r -- * "$DEST/"
mv ../git_package/git vllm/.git
mv ../git_package/github vllm/.github
