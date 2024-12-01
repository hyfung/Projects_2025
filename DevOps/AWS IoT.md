# AWS IOT

## IoT Core

- Device and data collection
- Command and control devices

## Key Components

- Device Gateway
  - Entry point for IoT devices
  - Uses MQTT
  - Devices and clients talk to each other
- Message Broker
  - Messaging agent
  - Bidirectional communcation between devices and applications
  - Pub-sub model
- Security and Identity
  - Authentication and authorization
  - IoT certificates and policies
  - 3 types of identity
    - X.509 certificate
    - IAM Users
    - Cognito (Third party authentication)
- Device Registry
  - Database of devices and attributes
  - Optional

## Provisioning Devices

### Thing Name

- Must be unique in same AWS region
- Do not use colon
- Standardize your naming convention

### Thing Attributes

- Descriptive information
  - Function, location, tech specs, etc
- Name-value pairs
- List up to three attributes

### Thing Type

- Logical categories
- List up to 50 attributes
- Thing can only be associated with one type
- No number of thing types limitation
- Unique within account
- Immutable after creation

### Thing Groups

- Manage several things at once
- Set policies at organization level
- Hierarchy of groups
- A thing can be member up to 10 groups

```json
{
  "version": 3,
  "thingName": "truckSensor01",
  "defaultClientId": "truckSensor01",
  "thingTypeName": "sensor_DoorAndPower",
  "attributes": {
    "deviceId": "T001",
    "powerRequirements": "12v"
  }
}
```

```json
{
  "thingTypes": [
    {
      "thingTypeName": "sensor_DoorAndTemp",
      "thingTypeProperties": {
        "searchableAttributes": ["deviceId", "powerRequirements"],
        "thingTypeDescription": "sensor for freezer trucks"
      },
      "thingTypeMetadata": {
        "deprecated": false,
        "creationDate": 1468423800950
      }
    }
  ]
}
```

## Connecting a Device

- AWS CLI or AWS SDK or Web Portal
- Connect-one or Connect-multiple
- Connect-one is a wizard
  - Prepare device
  - Register and secure device
  - Choose platform and SDK
  - Download connection kit
  - Run connection kit

### Connect One Device Wizard

- Create a thing type
- Add attributes to thing type
- Create the thing representing the node
- Register device
- Choose platform and SDK such as Ubuntu Python
- Download console connection kit

## Authorization and Authentication of IoT Devices

- AWS IoT Cert and Policy
  - Devices and things and application
- IAM Roles and Policy
  - Software and human operators

### Device Authentication

- Server authentiction
  - Device to authenticate the server
- Client authentication
  - Authenticate devices from IoT Core

Security Credentials

- X509 certificate to authenticate device
- IoT policy, attached to certificate for device authorization

Authentication by

- MQTT over TLS
- SigV4 over HTTP
- MQTT over WebSocket

### Device Authorization

IoT policies
- Grants permission to certificate

## References

[](https://cloud.contentraven.com/Embedded?oid=qddMi73E6Fhc/38Xw8a0xQ__&cid=OQmH+PDiBoM_)

[AWS IOT Fleet Indexing](https://www.coursera.org/learn/aws-managing-aws-iot-devices-fleet-indexing)
