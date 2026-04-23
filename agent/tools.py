def mock_lead_capture(name: str, email: str, platform: str) -> str:
    result = f"Lead captured successfully: {name}, {email}, {platform}"
    print(result)
    return result
