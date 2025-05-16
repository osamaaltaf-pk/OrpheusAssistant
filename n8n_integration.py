import json
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import jwt
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('n8n_integration')

class N8NIntegration:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'X-N8N-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data if data else None
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to n8n: {e}")
            raise

class ToolCaller:
    def __init__(self, n8n: N8NIntegration):
        self.n8n = n8n
        self.tools: Dict[str, Dict] = {}
        self._load_available_tools()
        
    def _load_available_tools(self):
        """Load available workflows as tools from n8n"""
        try:
            workflows = self.n8n._make_request('GET', '/workflows')
            for workflow in workflows:
                if workflow.get('active'):
                    self.tools[workflow['name']] = {
                        'id': workflow['id'],
                        'description': workflow.get('description', ''),
                        'parameters': self._extract_parameters(workflow)
                    }
            logger.info(f"Loaded {len(self.tools)} tools from n8n")
        except Exception as e:
            logger.error(f"Error loading tools: {e}")
            
    def _extract_parameters(self, workflow: Dict) -> List[Dict]:
        """Extract required parameters from workflow nodes"""
        parameters = []
        try:
            nodes = workflow.get('nodes', [])
            for node in nodes:
                if node.get('type') == 'n8n-nodes-base.function':
                    params = node.get('parameters', {}).get('functionCode', '')
                    # Extract parameters from function code (basic implementation)
                    if 'parameters' in params:
                        param_lines = [line for line in params.split('\n') if 'parameters.' in line]
                        for line in param_lines:
                            param_name = line.split('parameters.')[1].split()[0].strip()
                            parameters.append({
                                'name': param_name,
                                'type': 'string'  # Default type
                            })
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
        return parameters

    def list_tools(self) -> List[Dict]:
        """Return list of available tools and their descriptions"""
        return [
            {
                'name': name,
                'description': info['description'],
                'parameters': info['parameters']
            }
            for name, info in self.tools.items()
        ]

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict:
        """Call an n8n workflow with parameters"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        try:
            workflow_id = self.tools[tool_name]['id']
            endpoint = f'/workflows/{workflow_id}/execute'
            
            # Prepare execution data
            execution_data = {
                'workflowData': {
                    'id': workflow_id,
                    'name': tool_name
                },
                'parameters': parameters
            }
            
            # Execute workflow
            result = self.n8n._make_request('POST', endpoint, execution_data)
            
            return {
                'tool': tool_name,
                'status': 'success',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                'tool': tool_name,
                'status': 'error',
                'error': str(e)
            }

class CalendarTool:
    def __init__(self, tool_caller: ToolCaller):
        self.tool_caller = tool_caller
        
    async def create_event(self, title: str, start_time: datetime, end_time: datetime, description: str = "") -> Dict:
        parameters = {
            'title': title,
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'description': description
        }
        return await self.tool_caller.call_tool('create_calendar_event', parameters)
        
    async def list_events(self, start_date: datetime, end_date: datetime) -> Dict:
        parameters = {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        }
        return await self.tool_caller.call_tool('list_calendar_events', parameters)

class AppointmentTool:
    def __init__(self, tool_caller: ToolCaller):
        self.tool_caller = tool_caller
        
    async def schedule_appointment(self, title: str, date: datetime, duration_minutes: int, attendees: List[str]) -> Dict:
        parameters = {
            'title': title,
            'date': date.isoformat(),
            'duration': duration_minutes,
            'attendees': attendees
        }
        return await self.tool_caller.call_tool('schedule_appointment', parameters)
        
    async def cancel_appointment(self, appointment_id: str) -> Dict:
        parameters = {
            'appointment_id': appointment_id
        }
        return await self.tool_caller.call_tool('cancel_appointment', parameters)

class EmailTool:
    def __init__(self, tool_caller: ToolCaller):
        self.tool_caller = tool_caller
        
    async def send_email(self, to: List[str], subject: str, body: str) -> Dict:
        parameters = {
            'to': to,
            'subject': subject,
            'body': body
        }
        return await self.tool_caller.call_tool('send_email', parameters)

class ReminderTool:
    def __init__(self, tool_caller: ToolCaller):
        self.tool_caller = tool_caller
        
    async def set_reminder(self, title: str, due_date: datetime, description: str = "") -> Dict:
        parameters = {
            'title': title,
            'due_date': due_date.isoformat(),
            'description': description
        }
        return await self.tool_caller.call_tool('set_reminder', parameters)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize n8n integration
        n8n = N8NIntegration(
            base_url="http://localhost:5678",
            api_key="your_n8n_api_key"
        )
        
        # Create tool caller
        tool_caller = ToolCaller(n8n)
        
        # Initialize tools
        calendar = CalendarTool(tool_caller)
        appointments = AppointmentTool(tool_caller)
        email = EmailTool(tool_caller)
        reminders = ReminderTool(tool_caller)
        
        # List available tools
        print("Available tools:")
        for tool in tool_caller.list_tools():
            print(f"- {tool['name']}: {tool['description']}")
            
        # Example: Create calendar event
        event_result = await calendar.create_event(
            title="Team Meeting",
            start_time=datetime.now(),
            end_time=datetime.now(),
            description="Weekly team sync"
        )
        print("Calendar event result:", event_result)
        
    asyncio.run(main()) 