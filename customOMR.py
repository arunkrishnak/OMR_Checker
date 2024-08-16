import os
import re
import argparse
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


import re

def replace_paths_with_circles(svg_file, output_file, digits):
    with open(svg_file, 'r', encoding='utf-8') as file:
        svg_content = file.read()

    # Define the pattern for the 14 spaces and the replacement with digits separated by 2 spaces
    script_version_pattern = re.compile(r'2  2  2  2  2  2  2  2  2  2  2  2  2  2')
    # Create a version string with digits separated by 2 spaces
    version_string = '  '.join(digits)
    svg_content = script_version_pattern.sub(version_string, svg_content)

    initial_base_id = 14827  # ID of the first bubble

    for column_index, digit in enumerate(digits):
        row = int(digit)
        column = column_index
        
        target_id = initial_base_id + (row * 14 * 8) + (column * 8)
        target_id = 'use' + str(target_id)  # use14827
        
        pattern = re.compile(
            r'(<g[^>]*id=["\']{}["\'][^>]*>.*?)<path[^>]*>(.*?)</g>'.format(target_id),
            re.DOTALL
        )
        replacement = r'\1<circle cx="4.35" cy="-4.2" r="4.6" fill="black" /> </g>\2'
        svg_content = pattern.sub(replacement, svg_content)

    svg_content = svg_content.replace('fill-opacity="0"', 'fill-opacity="1"')

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(svg_content)

def insert_svg_logo(input_file, logo_file, output_file):
    x1, y1 = 7, 145
    x2, y2 = 100, 170

    width = x2 - x1
    height = y2 - y1

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_svg = BeautifulSoup(f, 'xml')

        with open(logo_file, 'r', encoding='utf-8') as f:
            logo_svg = BeautifulSoup(f, 'xml')

        logo_svg_elem = logo_svg.find('svg')
        if not logo_svg_elem:
            print(f"Error: The SVG logo file does not contain a valid <svg> element.")
            return

        logo_width = parse_dimension(logo_svg_elem.get('width', '100px'))
        logo_height = parse_dimension(logo_svg_elem.get('height', '100px'))

        scale = min(width / logo_width, height / logo_height)

        new_logo = logo_svg_elem
        new_logo['width'] = str(logo_width * scale) + 'px'
        new_logo['height'] = str(logo_height * scale) + 'px'
        new_logo['x'] = str(x1)
        new_logo['y'] = str(y1)

        for elem in new_logo.find_all(True):
            elem.attrs = {k.replace('{http://www.w3.org/2000/svg}', ''): v for k, v in elem.attrs.items()}

        input_svg.svg.append(new_logo)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(input_svg))

        print(f"SVG file '{input_file}' successfully processed. Modified file saved as '{output_file}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

def parse_dimension(dim_str):
    try:
        return float(dim_str.replace('px', '').replace('em', '').replace('rem', ''))
    except ValueError:
        raise ValueError(f"Unable to parse dimension '{dim_str}'")

def load_translations(translation_file):
    translations = {}
    try:
        with open(translation_file, "r", encoding="utf-8") as file:
            for line in file:
                if ':' in line:
                    key, value = line.split(':', 1)
                    translations[key] = value
    except IOError as e:
        print(f"IOError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return translations

def replace_text(input_file, translations, output_file):
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return

    try:
        with open(input_file, "r", encoding="utf-8") as fin:
            data = fin.read()

        for search, replace in translations.items():
            data = data.replace(search, replace)

        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(data)

        print(f"Successfully updated '{output_file}'.")

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
    except IOError as e:
        print(f"IOError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def convert_svg_to_pdf(svg_file, pdf_file):
    # Convert SVG to ReportLab Drawing object
    drawing = svg2rlg(svg_file)
    
    # Create a PDF file with ReportLab using A4 pagesize
    c = canvas.Canvas(pdf_file, pagesize=A4)
    page_width, page_height = A4

    # Get SVG dimensions
    svg_width = drawing.width
    svg_height = drawing.height

    # Keep SVG dimensions same (no scaling)
    drawing.scale(1, 1)

    # Calculate offsets to center the SVG on the page
    x_offset = (page_width - svg_width) / 2
    y_offset = (page_height - svg_height) / 2

    # Render the SVG drawing into the PDF
    renderPDF.draw(drawing, c, x_offset, y_offset)
    
    # Save the PDF
    c.save()


def process_multiple_files(base_command, svg_file, output_file, script_version, count, logo_file=None, translation_file=None):
    for i in range(count):
        version = str(int(script_version) + i)
        temp_file = f'temp_{version}.svg'
        temp_translated = f'temp_translated_{version}.svg'
        final_output = f'{output_file}_{version}.pdf'

        if base_command == 'prefillOMR':
            replace_paths_with_circles(svg_file, temp_file, version)
            convert_svg_to_pdf(temp_file, final_output)
            os.remove(temp_file)
        
        elif base_command == 'combine':
            replace_paths_with_circles(svg_file, temp_file, version)
            translations = load_translations(translation_file)
            replace_text(temp_file, translations, temp_translated)
            insert_svg_logo(temp_translated, logo_file, temp_translated)
            convert_svg_to_pdf(temp_translated, final_output)

            os.remove(temp_file)
            os.remove(temp_translated)

def main():
    parser = argparse.ArgumentParser(description="Process SVG files with various operations.")
    subparsers = parser.add_subparsers(dest='command')

    parser_replace = subparsers.add_parser('prefillOMR', help='Replace <path> elements with <circle> elements.')
    parser_replace.add_argument('svg_file', type=str, help='Path to the input SVG file.')
    parser_replace.add_argument('output_file', type=str, help='Path to the output PDF file prefix.')
    parser_replace.add_argument('script_version', type=str, help='Script version number consisting of 1 to 14 digits.')
    parser_replace.add_argument('count', type=int, help='Number of files to generate.')

    parser_logo = subparsers.add_parser('insertLogo', help='Insert an SVG logo into an SVG file.')
    parser_logo.add_argument('input_file', type=str, help='Input SVG file to process')
    parser_logo.add_argument('logo_file', type=str, help='SVG file containing the logo to insert')
    parser_logo.add_argument('output_file', type=str, help='Output SVG file to save the modified content')

    parser_replace_text = subparsers.add_parser('replaceText', help='Replace text in an SVG file using translations from another file.')
    parser_replace_text.add_argument('input_file', type=str, help='The path to the input SVG file to be translated.')
    parser_replace_text.add_argument('translation_file', type=str, help='The path to the file containing translations.')
    parser_replace_text.add_argument('output_file', type=str, help='The path to save the translated SVG file.')

    parser_combine = subparsers.add_parser('combine', help='Perform combined operations.')
    parser_combine.add_argument('input_file', type=str, help='The input SVG file.')
    parser_combine.add_argument('translation_file', type=str, help='File containing translations.')
    parser_combine.add_argument('logo_file', type=str, help='SVG file containing the logo to insert')
    parser_combine.add_argument('script_version', type=str, help='Script version number consisting of 1 to 14 digits.')
    parser_combine.add_argument('count', type=int, help='Number of files to generate.')
    parser_combine.add_argument('output_file', type=str, help='The path to save the final output PDF file prefix.')

    args = parser.parse_args()

    if args.command == 'prefillOMR':
        process_multiple_files('prefillOMR', args.svg_file, args.output_file, args.script_version, args.count)
    elif args.command == 'insertLogo':
        insert_svg_logo(args.input_file, args.logo_file, args.output_file)
    elif args.command == 'replaceText':
        translations = load_translations(args.translation_file)
        replace_text(args.input_file, translations, args.output_file)
    elif args.command == 'combine':
        process_multiple_files('combine', args.input_file, args.output_file, args.script_version, args.count, args.logo_file, args.translation_file)

if __name__ == "__main__":
    main()
