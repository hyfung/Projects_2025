# Warcraft 3

## Custom Kick

## How Multiplayer Works In General

- Uses 6112 TCP/UDP
- Should probably use wireshark to inspect packets
- Decompile `war3.exe` and inspect networking related functions

- Host collects all input from players
- Summarizes all the input
- Broadcast to all players
- Ensuring consistency and synchronization
- Not whole game state is transmitted, only inputs

## Map Editing Protector
