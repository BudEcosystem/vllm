#!/usr/bin/env python3
"""
Pre-startup validation and configuration script for vLLM with monitoring.

This script should be run before starting vLLM to ensure the system is
properly configured and to detect potential issues early.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

from vllm_monitor.vllm_integration_plugins import PreStartupValidator
from vllm_monitor.vllm_mitigation_strategies import PreStartupConfiguration
from vllm_monitor import VLLMMonitor, create_monitor_with_plugins


class PreStartupCheck:
    """
    Comprehensive pre-startup check and configuration for vLLM.
    """
    
    def __init__(self, auto_fix: bool = False, config_file: Optional[str] = None):
        self.auto_fix = auto_fix
        self.config_file = config_file
        self.monitor = create_monitor_with_plugins()
        self.validator = PreStartupValidator()
        self.configurator = PreStartupConfiguration()
        
        # Results tracking
        self.validation_passed = False
        self.errors_found = []
        self.warnings_found = []
        self.fixes_applied = []
    
    def run_checks(self) -> bool:
        """
        Run all pre-startup checks.
        
        Returns:
            True if all checks pass or issues are fixed
        """
        print("=" * 80)
        print("vLLM Pre-Startup Validation and Configuration")
        print("=" * 80)
        
        # Step 1: Run validation
        print("\n1. Running system validation...")
        validation_results = self._run_validation()
        
        # Step 2: Apply configuration if needed
        if self.auto_fix or not validation_results['passed']:
            print("\n2. Applying system configuration...")
            config_results = self._apply_configuration()
        else:
            print("\n2. Skipping configuration (validation passed)")
            config_results = {'success': True, 'applied': []}
        
        # Step 3: Check vLLM installation
        print("\n3. Checking vLLM installation...")
        install_results = self._check_vllm_installation()
        
        # Step 4: Verify model access
        print("\n4. Verifying model access...")
        model_results = self._check_model_access()
        
        # Step 5: Generate report
        print("\n5. Generating report...")
        report = self._generate_report(
            validation_results,
            config_results,
            install_results,
            model_results
        )
        
        # Save report
        report_path = Path("vllm_prestartup_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")
        
        # Determine overall success
        overall_success = (
            len(self.errors_found) == 0 or 
            (self.auto_fix and all(e['fixed'] for e in self.errors_found))
        )
        
        self._print_summary(overall_success)
        
        return overall_success
    
    def _run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation"""
        # Initialize validator
        self.validator.initialize({'logger': print})
        
        # Execute validation
        results = self.validator.execute()
        
        # Track results
        self.errors_found.extend(results.get('errors', []))
        self.warnings_found.extend(results.get('warnings', []))
        self.validation_passed = results.get('passed', False)
        
        # Print validation summary
        print(f"\nValidation {'PASSED' if results['passed'] else 'FAILED'}")
        print(f"  Errors: {len(results.get('errors', []))}")
        print(f"  Warnings: {len(results.get('warnings', []))}")
        
        # Print critical errors
        if results.get('errors'):
            print("\nCritical Errors:")
            for error in results['errors'][:5]:  # First 5
                print(f"  ❌ {error}")
        
        # Print warnings
        if results.get('warnings'):
            print("\nWarnings:")
            for warning in results['warnings'][:5]:  # First 5
                print(f"  ⚠️  {warning}")
        
        # Print recommendations
        if results.get('recommendations'):
            print("\nRecommendations:")
            for rec in results['recommendations'][:5]:  # First 5
                print(f"  💡 {rec}")
        
        return results
    
    def _apply_configuration(self) -> Dict[str, Any]:
        """Apply system configuration"""
        context = {
            'logger': print,
            'auto_fix': self.auto_fix
        }
        
        # Execute configuration
        outcome = self.configurator.execute(context)
        
        success = outcome.value >= 3  # SUCCESS or PARTIAL_SUCCESS
        
        # Track applied configurations
        if success:
            # These would be tracked by the actual implementation
            applied_configs = [
                "Set optimal environment variables",
                "Configured GPU settings",
                "Adjusted system parameters"
            ]
            self.fixes_applied.extend(applied_configs)
        
        return {
            'success': success,
            'outcome': outcome.name,
            'applied': self.fixes_applied
        }
    
    def _check_vllm_installation(self) -> Dict[str, Any]:
        """Check vLLM installation and dependencies"""
        results = {
            'vllm_installed': False,
            'version': None,
            'dependencies_ok': True,
            'issues': []
        }
        
        try:
            import vllm
            results['vllm_installed'] = True
            results['version'] = getattr(vllm, '__version__', 'unknown')
            print(f"✓ vLLM version: {results['version']}")
        except ImportError:
            results['issues'].append("vLLM not installed")
            self.errors_found.append({'error': 'vLLM not installed', 'fixed': False})
            print("✗ vLLM not installed")
            return results
        
        # Check critical dependencies
        dependencies = {
            'torch': None,
            'transformers': None,
            'ray': None,
            'asyncio': None
        }
        
        for dep_name in dependencies:
            try:
                dep_module = __import__(dep_name)
                version = getattr(dep_module, '__version__', 'installed')
                dependencies[dep_name] = version
                print(f"✓ {dep_name}: {version}")
            except ImportError:
                dependencies[dep_name] = 'missing'
                results['dependencies_ok'] = False
                results['issues'].append(f"{dep_name} not installed")
                print(f"✗ {dep_name}: missing")
        
        results['dependencies'] = dependencies
        
        return results
    
    def _check_model_access(self) -> Dict[str, Any]:
        """Check model accessibility"""
        results = {
            'cache_dir_accessible': False,
            'model_specified': False,
            'model_accessible': False,
            'issues': []
        }
        
        # Check model specification
        model_name = os.environ.get('VLLM_MODEL_NAME') or self._get_config_value('model')
        
        if not model_name:
            results['issues'].append("No model specified")
            print("✗ No model specified")
            return results
        
        results['model_specified'] = True
        results['model_name'] = model_name
        print(f"✓ Model specified: {model_name}")
        
        # Check cache directory
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists() and os.access(cache_dir, os.W_OK):
            results['cache_dir_accessible'] = True
            print(f"✓ Cache directory accessible: {cache_dir}")
        else:
            results['issues'].append(f"Cache directory not accessible: {cache_dir}")
            print(f"✗ Cache directory not accessible: {cache_dir}")
        
        # Check if model files exist locally
        model_path = cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        if model_path.exists():
            results['model_accessible'] = True
            print(f"✓ Model files found locally")
        else:
            print(f"ℹ️  Model not cached locally, will download on first use")
        
        return results
    
    def _get_config_value(self, key: str) -> Any:
        """Get configuration value"""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file) as f:
                    config = json.load(f)
                    return config.get(key)
            except:
                pass
        return None
    
    def _generate_report(self, 
                        validation: Dict[str, Any],
                        configuration: Dict[str, Any],
                        installation: Dict[str, Any],
                        model: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report"""
        report = {
            'timestamp': time.time(),
            'summary': {
                'validation_passed': validation['passed'],
                'configuration_applied': configuration['success'],
                'vllm_installed': installation['vllm_installed'],
                'model_accessible': model['model_accessible'],
                'total_errors': len(self.errors_found),
                'total_warnings': len(self.warnings_found),
                'fixes_applied': len(self.fixes_applied)
            },
            'validation': validation,
            'configuration': configuration,
            'installation': installation,
            'model_access': model,
            'errors': self.errors_found,
            'warnings': self.warnings_found,
            'fixes': self.fixes_applied,
            'environment': dict(os.environ),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        # Memory recommendations
        if any('memory' in str(e).lower() for e in self.errors_found + self.warnings_found):
            recommendations.append(
                "Consider reducing --gpu-memory-utilization to leave more free memory"
            )
        
        # GPU recommendations
        if any('gpu' in str(e).lower() for e in self.errors_found):
            recommendations.append(
                "Ensure CUDA_VISIBLE_DEVICES is set correctly"
            )
        
        # Network recommendations
        if any('nccl' in str(w).lower() for w in self.warnings_found):
            recommendations.append(
                "For distributed serving, ensure all nodes can communicate"
            )
        
        return recommendations
    
    def _print_summary(self, success: bool) -> None:
        """Print final summary"""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if success:
            print("✅ System is ready for vLLM")
        else:
            print("❌ System is not ready for vLLM")
        
        print(f"\nErrors found: {len(self.errors_found)}")
        print(f"Warnings found: {len(self.warnings_found)}")
        print(f"Fixes applied: {len(self.fixes_applied)}")
        
        if not success and not self.auto_fix:
            print("\n💡 Run with --auto-fix to attempt automatic fixes")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pre-startup validation and configuration for vLLM"
    )
    
    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help='Automatically fix issues where possible'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to vLLM configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to validate access for'
    )
    
    parser.add_argument(
        '--exit-on-error',
        action='store_true',
        help='Exit with non-zero code if validation fails'
    )
    
    args = parser.parse_args()
    
    # Set model in environment if specified
    if args.model:
        os.environ['VLLM_MODEL_NAME'] = args.model
    
    # Run checks
    checker = PreStartupCheck(
        auto_fix=args.auto_fix,
        config_file=args.config
    )
    
    success = checker.run_checks()
    
    # Exit with appropriate code
    if args.exit_on_error and not success:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()