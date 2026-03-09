# Context Management

The Scaleout CLI now supports managing multiple contexts, allowing you to easily switch between different Scaleout instances (e.g., development, staging, production).

## Features

- **Multiple named contexts**: Store credentials for multiple Scaleout instances
- **Quick switching**: Switch contexts by name or index number
- **Visual indicators**: See which context is currently active with a ★ marker
- **Automatic naming**: Contexts are automatically named based on the instance URL
- **Backward compatible**: Works seamlessly with existing setups

## Usage

### Login (creates a new context)

```bash
scaleout login https://production.example.com
# ✅ Logged in successfully to https://production.example.com
# 🔐 Access token stored securely.
```

Each time you login, a new context is created (or updated if it already exists).

### List all contexts

```bash
scaleout context
```

**Output:**
```
Available contexts:

  ★ [1] localhost:8092
      Host: http://localhost:8092
    [2] production.example.com
      Host: https://production.example.com
    [3] staging.example.com
      Host: https://staging.example.com

★ = active context
```

### Switch context by name

```bash
scaleout context production.example.com
# ✅ Switched to context 'production.example.com'
#    Host: https://production.example.com
```

### Switch context by index (quick!)

```bash
scaleout context 2
# ✅ Switched to context 'production.example.com'
#    Host: https://production.example.com
```

This is especially useful when you have many contexts and want to switch quickly.

### View current context

When running any command that connects to Scaleout, the CLI will log which context it's using:

```bash
scaleout model list
# INFO: Using host: 'https://production.example.com' from context 'production.example.com'
```

## File Structure

Contexts are stored in `~/.scaleout/`:

- `contexts.yaml` - All saved contexts with their credentials
- `active.yaml` - Tracks which context is currently active
- `context.yaml` - Legacy file for backward compatibility (contains active context)

## Improvements Over Original Design

1. **Named contexts**: Instead of just switching by index, contexts have meaningful names
2. **More information**: Shows both the context name and host URL
3. **Better error handling**: Clear error messages when switching to non-existent contexts
4. **Sorted display**: Contexts are always displayed in the same order for consistency
5. **Range validation**: Prevents invalid index selections
6. **Backward compatibility**: Existing code continues to work without changes
