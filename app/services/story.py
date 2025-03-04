import openai
import os
import logging
from typing import List, Dict, Optional
from app.models import PanelInfo, Story, Panel

logger = logging.getLogger(__name__)

class StoryService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if not api_key:
            logging.warning("No API key provided for story service")
        else:
            openai.api_key = self.api_key

    async def generate_story(self, title: str, num_panels: int) -> Story:
        if not self.api_key:
            # Return default panels if no API key
            return Story(
                title=title,
                panels=[Panel(prompt=f"Panel {i+1}: {title}") for i in range(num_panels)]
            )
        """Generate a story using OpenAI's GPT model"""
        system_prompt = (
            "You are a creative assistant that helps generate short comic stories."
        )

        user_prompt = f"""
        Create a short comic story titled "{title}" divided into {num_panels} panels.
        Ensure that each panel logically follows the previous one, maintains character consistency,
        and includes specific details to aid in image generation.

        For each panel, provide both a **Title** and a **Description**.

        Format:
        Panel 1:
        Title: [Title for Panel 1]
        Description: [Description for Panel 1]

        Panel 2:
        Title: [Title for Panel 2]
        Description: [Description for Panel 2]

        ...

        Panel {num_panels}:
        Title: [Title for Panel {num_panels}]
        Description: [Description for Panel {num_panels}]
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1200,
                temperature=0.7,
                n=1,
                stop=None
            )
            story_text = response.choices[0].message['content'].strip()
            panels = story_text.split('\n\n')
            story_panels = []
            
            for panel in panels:
                if not panel.strip():
                    continue
                try:
                    lines = panel.split('\n')
                    panel_num_line = lines[0].strip()
                    title_line = lines[1].strip()
                    description_line = lines[2].strip()

                    panel_num = panel_num_line.replace("Panel", "").replace(":", "").strip()
                    title = title_line.replace("Title:", "").strip()
                    description = description_line.replace("Description:", "").strip()
                    detailed_prompt = f"{title}. {description}, in a prehistoric Flintstones style."

                    story_panels.append(Panel(
                        prompt=detailed_prompt
                    ))
                except IndexError:
                    logger.warning(f"Could not parse panel:\n{panel}\n")
                    continue
                
            return Story(
                title=title,
                panels=story_panels
            )
        except Exception as e:
            logger.error(f"Error generating story: {str(e)}")
            raise

    def create_image_prompts(self, story_text: str, style: str = "prehistoric Flintstones style") -> List[PanelInfo]:
        """Parse the generated story and create image prompts"""
        panels = story_text.split('\n\n')
        prompts = []
        
        for panel in panels:
            if not panel.strip():
                continue
            try:
                lines = panel.split('\n')
                panel_num_line = lines[0].strip()
                title_line = lines[1].strip()
                description_line = lines[2].strip()

                panel_num = panel_num_line.replace("Panel", "").replace(":", "").strip()
                title = title_line.replace("Title:", "").strip()
                description = description_line.replace("Description:", "").strip()
                detailed_prompt = f"{title}. {description}, in a {style}."

                prompts.append(PanelInfo(
                    panel_num=panel_num,
                    title=title,
                    description=description,
                    image_prompt=detailed_prompt
                ))
            except IndexError:
                logger.warning(f"Could not parse panel:\n{panel}\n")
                continue
                
        return prompts 