"""
Command-line interface for APA.

Provides CLI commands for running pipelines, validating configurations,
and managing the APA system.
"""

import click
import sys
from pathlib import Path

from apa.config.manager import ConfigManager
from apa.pipeline.runner import APAPipeline


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """APA - Advanced Pavement Analytics CLI."""
    pass


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to configuration file')
def run(config: str):
    """Run the APA pipeline with a configuration file."""
    try:
        config_manager = ConfigManager()
        config_data = config_manager.load_config(config)
        
        click.echo(f"Loading configuration from: {config}")
        pipeline = APAPipeline(config_data['config'])
        
        click.echo("Running APA pipeline...")
        result = pipeline.run()
        
        if result.success:
            click.echo(f"✓ Pipeline completed successfully in {result.execution_time:.2f}s")
        else:
            click.echo(f"✗ Pipeline completed with errors:")
            for error in result.errors:
                click.echo(f"  - {error}")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to configuration file')
def validate(config: str):
    """Validate a configuration file."""
    try:
        config_manager = ConfigManager()
        config_data = config_manager.load_config(config)
        
        is_valid = config_manager.validate_config(config_data['config'])
        
        if is_valid:
            click.echo(f"✓ Configuration file is valid: {config}")
        else:
            click.echo(f"✗ Configuration file is invalid: {config}", err=True)
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--template', '-t', default='default',
              help='Template name')
@click.option('--output', '-o', required=True,
              help='Output path for configuration file')
def create_config(template: str, output: str):
    """Create a new configuration file from a template."""
    try:
        config_manager = ConfigManager()
        output_path = config_manager.create_config_from_template(template, output)
        
        click.echo(f"✓ Configuration file created: {output_path}")
    
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def list_templates():
    """List available configuration templates."""
    click.echo("Available templates:")
    click.echo("  - default: Default APA pipeline configuration")
    click.echo("  - detroit: Detroit-specific configuration")
    click.echo("  - venus_israel: VENUS Israel configuration")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
def info(config: Optional[str]):
    """Show APA system information."""
    click.echo("APA - Advanced Pavement Analytics")
    click.echo("Version: 0.1.0")
    click.echo("")
    
    if config:
        try:
            config_manager = ConfigManager()
            config_data = config_manager.load_config(config)
            click.echo(f"Configuration: {config}")
            click.echo(f"  Sections: {', '.join(config_data['config'].keys())}")
        except Exception as e:
            click.echo(f"Error loading config: {str(e)}", err=True)


if __name__ == '__main__':
    cli()

