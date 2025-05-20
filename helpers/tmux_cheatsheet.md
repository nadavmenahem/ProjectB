# Start a new session
```bash
tmux new -s session_name
```

# Enter a session
```bash
tmux attach -t mysession
```

# Exit a session
```bash
Ctrl-b d
```

# List existing sessions
```bash
tmux ls
```

# Delete a session
```bash
tmux kill-session -t mysession
```