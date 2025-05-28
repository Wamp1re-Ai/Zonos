# Using Cloudflare Subdomains with Zonos Gradio Interface

This guide explains how to use a custom Cloudflare subdomain with your Zonos Gradio interface.

## Overview

The Gradio interface for Zonos has been updated to support custom Cloudflare subdomains. This allows you to access your Gradio interface via a consistent URL of the form `https://your-subdomain.gradio.app`.

## Usage Instructions

### Running Locally

To run the Gradio interface with a custom subdomain:

```bash
# Replace 'my-zonos-app' with your preferred subdomain name
GRADIO_SUBDOMAIN=my-zonos-app GRADIO_SHARE=true python gradio_interface.py
```

### In Google Colab

A cell has been added to the Colab notebook that shows how to set the subdomain:

```python
import os

# Set your desired subdomain name
subdomain_name = "my-zonos-app"  # Change this to your preferred subdomain name

# Set environment variables
os.environ["GRADIO_SUBDOMAIN"] = subdomain_name
os.environ["GRADIO_SHARE"] = "True"

# Run the gradio interface
!python gradio_interface.py
```

## Important Notes

1. **Subdomain Availability**: Subdomain names are allocated on a first-come, first-served basis. If your chosen subdomain is already in use, you'll need to select a different one.

2. **Subdomain Duration**: Cloudflare subdomains are typically reserved for 72 hours after last use. After this period, they may become available to other users.

3. **Authentication**: These public URLs do not include authentication. If you need more security, consider using Gradio's built-in authentication features or hosting the interface behind a proper authentication system.

4. **Colab Limitations**: Remember that Colab sessions time out after a period of inactivity. For long-term hosting, consider deploying to a dedicated server.
