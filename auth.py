import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, send_file, current_app, abort
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename
import re
from extensions import db
from models import UserCredentials, TeacherDetails, StudentDetails, AdminProfile, HODProfile, SubjectAssignment, Course, \
    CourseSubject, Attendance, BatchTable, OTPModel
import random
import string
import logging
from datetime import datetime,timedelta
from sqlalchemy import inspect, or_, text
import csv
from io import StringIO
import imghdr
import hashlib, time
import pandas as pd
from flask import  send_from_directory
import io
from flask import Response, json, stream_with_context
from flask import session
from sqlalchemy.exc import SQLAlchemyError
import smtplib
from email.mime.text import MIMEText
import random
import string
from flask_mail import Message
from extensions import mail

bp = Blueprint('auth', __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Set up logging
logger = logging.getLogger(__name__)

def generate_secure_filename(filename):
    ext = filename.split(".")[-1]
    hashed_name = hashlib.sha256(f"{filename}{time.time()}".encode()).hexdigest()[:20]
    return f"{hashed_name}.{ext}"





@bp.route("/protected_image/<folder>/<filename>")
@login_required
def protected_image(folder, filename):
    UPLOAD_FOLDER = "uploads"  # Base upload folder (outside static)
    # ✅ Only allow specific folders
    allowed_folders = ["student_photos", "hod_photos", "teacher_photos"]
    if folder not in allowed_folders:
        abort(403)  # Forbidden

    file_path = os.path.join(UPLOAD_FOLDER, folder, filename)

    # ✅ Check if file exists
    if not os.path.exists(file_path):
        abort(404)  # File Not Found

    # ✅ Enforce Role-Based Access Control
    if folder == "student_photos" and current_user.role not in ["admin", "student"]:
        abort(403)  # Only students & admin can access

    if folder == "hod_photos" and current_user.role not in ["admin", "hod"]:
        abort(403)  # Only HOD & admin can access

    if folder == "teacher_photos" and current_user.role not in ["admin", "teacher"]:
        abort(403)  # Only teachers & admin can access

    return send_file(file_path)


def generate_random_password(length=12):
    """Generate a random password with letters, digits, and special characters"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choice(characters) for _ in range(length))


def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

@bp.route('/forgot_password_page', methods=['GET'])
def forgot_password_page():
    show_otp_form = request.args.get('show_otp_form', False)
    email = request.args.get('email', '')
    return render_template('forgot_password.html', show_otp_form=show_otp_form, email=email)

@bp.route('/send_otp', methods=['POST'])
def send_otp():
    try:
        email = request.form.get('email')
        if not email:
            flash('Email is required')
            return redirect(url_for('auth.forgot_password_page'))
        
        # Check if user exists
        user = UserCredentials.query.filter_by(email=email).first()
        if not user:
            flash('No account found with this email')
            return redirect(url_for('auth.forgot_password_page'))
        
        # Generate and save OTP
        otp = generate_otp()
        
        # Delete any existing OTPs for this email
        OTPModel.query.filter_by(email=email).delete()
        db.session.commit()
        
        # Create new OTP record
        otp_record = OTPModel(
            email=email,
            otp_code=otp,
            expires_at=datetime.utcnow() + timedelta(minutes=10)
        )
        db.session.add(otp_record)
        db.session.commit()
        
        # Prepare HTML email
        html_body = render_template(
            'otp_email.html',  # Assuming you have a username field
            otp=otp
        )
        
        # Send OTP via email with HTML content
        msg = Message(
            'Password Reset OTP',
            recipients=[email],
            html=html_body,
        )
        mail.send(msg)
        
        flash('OTP has been sent to your email')
        return redirect(url_for('auth.forgot_password_page', show_otp_form=True, email=email))
    
    except Exception as e:
        logger.error(f"Error in send_otp: {str(e)}")
        flash('An error occurred while sending OTP')
        return redirect(url_for('auth.forgot_password_page'))

@bp.route('/verify_otp_and_reset', methods=['POST'])
def verify_otp_and_reset():
    try:
        email = request.form.get('email')
        otp = request.form.get('otp')

        if not email or not otp:
            flash('Both email and OTP are required')
            return redirect(url_for('auth.forgot_password_page'))

        # Find and verify OTP
        otp_record = OTPModel.query.filter_by(email=email, otp_code=otp, is_used=False).first()
        
        if not otp_record or not otp_record.is_valid():
            flash('Invalid or expired OTP')
            return redirect(url_for('auth.forgot_password_page', show_otp_form=True, email=email))

        # Generate new password
        new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        
        # Update password in database
        user = UserCredentials.query.filter_by(email=email).first()
        user.set_password(new_password)
        
        # Mark OTP as used
        otp_record.is_used = True
        db.session.commit()

        html_content = render_template('email_template.html', password=new_password)
        # Send new password via email
        msg = Message(
            'New Password',
            recipients=[email],
        )
        msg.html = html_content
        mail.send(msg)

        flash('Your password has been reset. Please check your email for the new password.')
        return redirect(url_for('auth.login'))

    except Exception as e:
        logger.error(f"Error in verify_otp_and_reset: {str(e)}")
        flash('An error occurred while resetting password')
        return redirect(url_for('auth.forgot_password_page'))



@bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))
    return redirect(url_for('auth.login'))


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')

        logging.info(f"Login attempt for email: {email}, role: {role}")

        if not all([email, password, role]):
            flash('Please fill in all required fields')
            return redirect(url_for('auth.login'))

        user = UserCredentials.query.filter_by(email=email, role=role).first()

        if not user:
            logging.warning(f"No user found with email: {email} and role: {role}")
            flash('Invalid credentials')
            return redirect(url_for('auth.login'))

        if not user.check_password(password):
            logging.warning(f"Invalid password for user: {email}")
            flash('Invalid credentials')
            return redirect(url_for('auth.login'))

        if not user.is_active:
            flash('Your account is deactivated. Please contact administrator.')
            return redirect(url_for('auth.login'))

        # Additional verification for students
        if role == 'student':
            course_id = request.form.get('course')
            admission_year = request.form.get('admission_year')
            semester = request.form.get('semester')
            batch_id = request.form.get('batch_id')

            if not all([course_id, admission_year, semester, batch_id]):
                flash('Please fill in all student details')
                return redirect(url_for('auth.login'))

            # Find the batch record to get the table_name
            batch = BatchTable.query.filter_by(
                course_id=course_id,
                admission_year=admission_year,
                semester=semester,
                batch_id=batch_id
            ).first()

            if not batch:
                flash('Invalid batch details provided')
                return redirect(url_for('auth.login'))

            # Get the table_name from the batch record
            table_name = batch.table_name

            # Use raw SQL to query the student in the dynamic table
            sql = f"""
                SELECT * FROM {table_name}
                WHERE credential_id = :credential_id
                AND email = :email
            """

            result = db.session.execute(
                text(sql),
                {
                    'credential_id': user.id,
                    'email': email
                }
            ).fetchone()

            if not result:
                flash('Your credentials do not match any student in the selected batch')
                return redirect(url_for('auth.login'))

            # Store batch information in session for later use
            session['batch_id'] = batch_id
            session['course_id'] = course_id
            session['admission_year'] = admission_year
            session['semester'] = semester
            session['batch_table_name'] = table_name

        login_user(user)
        logging.info(f"Successful login for user: {email}, role: {role}")
        return redirect(url_for(f'auth.{user.role}_dashboard'))

    return render_template('login.html')


@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('auth.login'))


# @bp.route('/add_user', methods=['POST'])
# @login_required
# def add_user():
#     print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#     if current_user.role != 'admin':
#         flash('Access denied: Admin privileges required')
#         return redirect(url_for(f'auth.{current_user.role}_dashboard'))
#
#     try:
#         role = request.form.get('role')
#         email = request.form.get('email')
#         first_name = request.form.get('first_name')
#         last_name = request.form.get('last_name')
#         phone = request.form.get('phone')
#         address = request.form.get('address')
#         department = request.form.get('department')
#
#         # Handle photo upload
#         photo_path = None
#         if 'photo' in request.files:
#             photo = request.files['photo']
#             if photo.filename:
#                 filename = f"{role}_{email.split('@')[0]}_{photo.filename}"
#                 photo_path = os.path.join('static', 'uploads', 'photos', filename)
#                 os.makedirs(os.path.dirname(photo_path), exist_ok=True)
#                 photo.save(photo_path)
#
#         # Generate random password
#         password = generate_random_password()
#
#         # Create credentials
#         credentials = UserCredentials(
#             email=email,
#             role=role
#         )
#         credentials.set_password(password)
#         db.session.add(credentials)
#         db.session.flush()  # Get the ID before creating profile
#
#         if role == 'student':
#             print("*******************************************************************************************************************************************")
#             admission_year = request.form.get('admission_year')
#             roll_number = request.form.get('roll_number')
#             course_id = request.form.get('course_id')
#
#             # Validate admission year exists in batches
#             available_years = get_available_batch_years(course_id)
#             if int(admission_year) not in available_years:
#                 flash('Selected admission year does not exist in batches')
#                 return redirect(url_for('auth.admin_dashboard'))
#
#             # Calculate current year based on admission year
#             current_year = datetime.utcnow().year - int(admission_year) + 1
#             if current_year < 1:
#                 current_year = 1
#             elif current_year > 4:
#                 current_year = 4
#
#             # Determine the batch table name
#             batch_table = f'student_batch_{course_id}_{admission_year}'
#
#             # Insert student into the batch table instead of StudentDetails
#             sql = text(f"""
#             INSERT INTO {batch_table} (
#                 credential_id, first_name, last_name, email, phone, address, photo_path,
#                 roll_number, current_year, current_semester, admission_year, course_id, batch, admission_date
#             ) VALUES (
#                 :credential_id, :first_name, :last_name, :email, :phone, :address, :photo_path,
#                 :roll_number, :current_year, :current_semester, :admission_year, :course_id, :batch, :admission_date
#             )
#             """)
#
#             db.session.execute(sql, {
#                 'credential_id': credentials.id,
#                 'first_name': first_name,
#                 'last_name': last_name,
#                 'email': email,
#                 'phone': phone,
#                 'address': address,
#                 'photo_path': photo_path,
#                 'roll_number': roll_number,
#                 'current_year': current_year,
#                 'current_semester': 1,
#                 'admission_year': int(admission_year),
#                 'course_id': course_id,
#                 'batch': str(admission_year),
#                 'admission_date': datetime.utcnow()
#             })
#
#         elif role == 'teacher':
#             print("#####################################################################################################################################################")
#             sql_teacher = text("""
#                 INSERT INTO teacher_details (
#                     credential_id, first_name, last_name, email, department, phone, address, photo_path
#                 ) VALUES (
#                     :credential_id, :first_name, :last_name, :email, :department, :phone, :address, :photo_path
#                 )
#             """)
#
#             db.session.execute(sql_teacher, {
#                 'credential_id': credentials.id,
#                 'first_name': first_name,
#                 'last_name': last_name,
#                 'email': email,
#                 'department': department,
#                 'phone': phone,
#                 'address': address,
#                 'photo_path': photo_path
#             })
#
#         elif role == 'hod':
#             sql_hod = text("""
#                 INSERT INTO hod_profiles (
#                     credential_id, first_name, last_name, email, phone, address, photo_path, department, office_location
#                 ) VALUES (
#                     :credential_id, :first_name, :last_name, :email, :phone, :address, :photo_path, :department, :office_location
#                 )
#             """)
#
#             db.session.execute(sql_hod, {
#                 'credential_id': credentials.id,
#                 'first_name': first_name,
#                 'last_name': last_name,
#                 'email': email,
#                 'phone': phone,
#                 'address': address,
#                 'photo_path': photo_path,
#                 'department': department,
#                 'office_location': request.form.get('office_location', 'Sanjay Nagar,Ghaziabad')
#             })
#
#         db.session.commit()
#         flash(f'User created successfully. Temporary password: {password}')
#         logging.info(f'New {role} user created: {email}')
#
#     except Exception as e:
#         db.session.rollback()
#         flash(f'Error creating user: {str(e)}')
#         logging.error(f'Error creating user: {str(e)}')
#
#     return redirect(url_for('auth.admin_dashboard'))


@bp.route('/create_student_batch', methods=['POST'])
@login_required
def create_student_batch():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required', 'danger')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    try:
        batch_year = request.form.get('batch_year')
        course_id = request.form.get('course_id')
        batch_id = request.form.get('batch_id')
        semester = request.form.get('semester')

        if not all([batch_year, course_id, batch_id, semester]):
            flash('Batch year, course, batch ID, and semester are required', 'warning')
            return redirect(url_for('auth.admin_dashboard'))

        # Validate batch year
        try:
            batch_year = int(batch_year)
            if batch_year < 1900 or batch_year > 2100:
                raise ValueError("Batch year out of valid range")
        except ValueError:
            flash('Invalid batch year format', 'danger')
            return redirect(url_for('auth.admin_dashboard'))

        # Validate batch ID (1 to 5)
        try:
            batch_id = int(batch_id)
            if batch_id < 1 or batch_id > 5:
                raise ValueError("Batch ID must be between 1 and 5")
        except ValueError:
            flash('Invalid batch ID', 'danger')
            return redirect(url_for('auth.admin_dashboard'))

        # Validate semester (1 to 8)
        try:
            semester = int(semester)
            if semester < 1 or semester > 8:
                raise ValueError("Semester must be between 1 and 8")
        except ValueError:
            flash('Invalid semester', 'danger')
            return redirect(url_for('auth.admin_dashboard'))

        # Fetch course to validate and use in table name
        course = Course.query.get(course_id)
        if not course:
            flash('Invalid course selected', 'danger')
            return redirect(url_for('auth.admin_dashboard'))

        # Generate table name
        table_name = f'student_batch_{course_id}_{batch_year}_{semester}_{batch_id}'

        # SQL Query to create the table
        create_table_sql = text(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            credential_id INT UNIQUE NOT NULL,
            first_name VARCHAR(64) NOT NULL,
            last_name VARCHAR(64) NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            phone VARCHAR(20),
            address TEXT,
            photo_path VARCHAR(500),
            roll_number VARCHAR(20) UNIQUE NOT NULL,
            current_year INT NOT NULL,
            admission_year INT NOT NULL,
            course_id INT NOT NULL,
            admission_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            batch_id INT NOT NULL,
            semester INT NOT NULL,
            FOREIGN KEY (credential_id) REFERENCES user_credentials(id) ON DELETE CASCADE,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        # Execute the table creation
        db.session.execute(create_table_sql)

        # Store the batch table information in the new table
        batch_table = BatchTable(
            table_name=table_name,
            course_id=course.id,
            course_name=course.name,  # Assuming your Course model has a 'name' attribute
            course_code=course.code,  # Assuming your Course model has a 'code' attribute
            admission_year=batch_year,
            semester=semester,
            batch_id=batch_id
        )

        db.session.add(batch_table)
        db.session.commit()

        flash(f'Student batch table `{table_name}` created successfully!', 'success')
        logging.info(f'Successfully created student batch table: {table_name}')

    except Exception as e:
        db.session.rollback()
        flash(f'Error creating student batch: {str(e)}', 'danger')
        logging.error(f'Error creating student batch: {str(e)}')

    return redirect(url_for('auth.admin_dashboard'))


@bp.route('/get_batch_years')
def get_batch_years():
    """API endpoint to get available batch years"""
    try:
        course_id = request.args.get('course_id')
        if not course_id:
            return jsonify({'error': 'Course ID is required'}), 400

        # Query to get all tables that are related to the specified course
        sql = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE :pattern
        AND table_schema = DATABASE()
        """)
        result = db.session.execute(sql, {'pattern': f'student_batch_{course_id}_%'})
        batch_tables = result.fetchall()

        # Extract years from table names and sort them
        years = []
        for (table_name,) in batch_tables:
            try:
                year = int(table_name.split('_')[-1])
                years.append(year)
            except ValueError:
                continue

        sorted_years = sorted(years, reverse=True)
        logging.info(f'Retrieved batch years for course {course_id}: {sorted_years}')
        return jsonify({'years': sorted_years})
    except Exception as e:
        logging.error(f'Error fetching batch years: {str(e)}')
        return jsonify({'error': str(e)}), 500


def get_available_batch_years(course_id=None):
    """Fetch all available batch years from existing batch tables for a specific course"""
    try:
        pattern = f'student_batch_{course_id}_%' if course_id else 'student_batch_%'
        # Query to get all tables that start with the pattern
        sql = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE :pattern
        AND table_schema = DATABASE()
        """)
        result = db.session.execute(sql, {'pattern': pattern})
        batch_tables = result.fetchall()

        # Extract years from table names and sort them
        years = []
        for (table_name,) in batch_tables:
            try:
                year = int(table_name.split('_')[-1])
                years.append(year)
            except ValueError:
                continue

        sorted_years = sorted(years, reverse=True)
        logging.info(f'Retrieved batch years for course {course_id}: {sorted_years}')
        return sorted_years
    except Exception as e:
        logging.error(f'Error fetching batch years: {str(e)}')
        return []


@bp.route('/add_hod', methods=['GET', 'POST'])
@login_required
def add_hod():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    if request.method == 'POST':
        try:
            email = request.form.get('email')
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            phone = request.form.get('phone')
            address = request.form.get('address')
            department = request.form.get('department')
            office_location = request.form.get('office_location', 'Sanjay Nagar, Ghaziabad')

            # Handle photo upload
            photo_path = None
            if 'photo' in request.files:
                photo = request.files['photo']
                if photo.filename:
                    filename = generate_secure_filename(photo.filename)
                    photo_paths = os.path.join('uploads', 'hod_photos', filename)
                    photo_path = filename
                    os.makedirs(os.path.dirname(photo_paths), exist_ok=True)
                    photo.save(photo_paths)

            # Generate random password
            password = generate_random_password()

            # Create credentials
            credentials = UserCredentials(email=email, role='hod')
            credentials.set_password(password)
            db.session.add(credentials)
            db.session.flush()

            # Insert into HOD table
            sql_hod = text("""
                INSERT INTO hod_profiles (
                    credential_id, first_name, last_name, email, phone, address, photo_path, department, office_location
                ) VALUES (
                    :credential_id, :first_name, :last_name, :email, :phone, :address, :photo_path, :department, :office_location
                )
            """)

            db.session.execute(sql_hod, {
                'credential_id': credentials.id,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': phone,
                'address': address,
                'photo_path': photo_path,
                'department': department,
                'office_location': office_location
            })

            db.session.commit()
            flash(f'HOD added successfully. Temporary password: {password}')
            return redirect(url_for('auth.render_hod'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error adding HOD: {str(e)}')

    return render_template('add_hod.html')


# Allowed image extensions


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    """Check if the uploaded file has a valid image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/add_teacher', methods=['GET', 'POST'])
@login_required
def add_teacher():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required', 'danger')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    if request.method == 'POST':
        try:
            email = request.form.get('email')
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            phone = request.form.get('phone')
            address = request.form.get('address')
            department = request.form.get('department')

            # **BACKEND VALIDATION: Check Email Format**
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                flash("Invalid email format!", "danger")
                return redirect(url_for('auth.add_teacher'))

            # **BACKEND VALIDATION: Ensure Phone Number is 10 Digits**
            if not phone.isdigit():
                flash("Phone number must contain only numbers!", "danger")
                return redirect(url_for('auth.add_teacher'))

            if len(phone) == 11 and phone.startswith("0"):
                phone = phone[1:]  # Remove first zero
            elif len(phone) != 10:
                flash("Phone number must be exactly 10 digits or 11 digits starting with 0!", "danger")
                return redirect(url_for('auth.add_teacher'))

            # Generate random password
            password = generate_random_password()

            # Create credentials
            credentials = UserCredentials(email=email, role='teacher')
            credentials.set_password(password)
            db.session.add(credentials)
            db.session.flush()  # Get credential_id before committing

            # Define default photo path
            photo_path = None

            # **BACKEND VALIDATION: Check and Save Image File**
            if 'photo' in request.files:
                photo = request.files['photo']
                if photo.filename:
                    if not allowed_file(photo.filename):
                        flash("Invalid file format! Only PNG, JPG, JPEG, GIF allowed.", "danger")
                        return redirect(url_for('auth.add_teacher'))

                    # Ensure upload folder exists
                    upload_folder = os.path.join('uploads', 'teacher_photos')
                    os.makedirs(upload_folder, exist_ok=True)

                    # Secure filename and save
                    photo_filename = generate_secure_filename(photo.filename)
                    photo_path = f"{photo_filename}"
                    photo.save(os.path.join(upload_folder, photo_filename))

            # Insert into Teacher table
            sql_teacher = text("""
                INSERT INTO teacher_details (
                    credential_id, first_name, last_name, email, department, phone, address, photo_path
                ) VALUES (
                    :credential_id, :first_name, :last_name, :email, :department, :phone, :address, :photo_path
                )
            """)

            db.session.execute(sql_teacher, {
                'credential_id': credentials.id,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'department': department,
                'phone': phone,
                'address': address,
                'photo_path': photo_path
            })

            db.session.commit()
            flash(f'Teacher added successfully. Temporary password: {password}', 'success')
            return redirect(url_for('auth.render_teacher'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error adding teacher: {str(e)}', 'danger')

    return render_template('dashboard/add_teacher.html')


@bp.route('/render_hod', methods=['GET', 'POST'])
@login_required
def render_hod():
    return render_template('dashboard/add_hod.html')


@bp.route('/render_teacher', methods=['GET', 'POST'])
@login_required
def render_teacher():
    return render_template('dashboard/add_teacher.html')


@bp.route('/render_student', methods=['GET', 'POST'])
@login_required
def render_student():
    result = db.session.execute(text("SHOW TABLES LIKE 'student_batch_%'"))
    tables = [row[0] for row in result.fetchall()]  # Fetch all batch tables

    courses = Course.query.all()
    course_subjects = CourseSubject.query.all()
    result = db.session.execute(text("SHOW TABLES LIKE 'student_batch_%'"))
    tables = [row[0] for row in result.fetchall()]  # Fetch all table names

    # Fetch all existing student batch data for dropdown handling

    return render_template('dashboard/add_student.html', courses=courses, table_names=tables)


@bp.route('/add_student', methods=['POST'])
@login_required
def add_student():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    try:
        # Fetch form data
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        phone = request.form.get('phone')
        address = request.form.get('address')
        admission_year = request.form.get('admission_year')
        roll_number = request.form.get('roll_number')
        course_id = request.form.get('course_id')
        batch_id = request.form.get('batch_id')  # Batch ID
        semester = request.form.get('semester')  # Semester selection

        # Ensure semester value is valid (1 to 8)
        semester = int(semester)
        if semester < 1 or semester > 8:
            flash('Invalid semester. Choose between 1 and 8.')
            return redirect(url_for('auth.admin_dashboard'))

        # Validate roll_number (must be an integer and exactly 10 digits)
        if not roll_number.isdigit() :
            flash("Invalid roll number")
            return redirect(url_for('auth.admin_dashboard'))

        # Validate phone number
        if phone.isdigit():
            if len(phone) == 11 and phone.startswith('0'):
                phone = phone[1:]  # Remove leading zero
            elif len(phone) == 13 and phone.startswith('+91'):
                phone = phone[3:]  # Remove "+91"

        if not phone.isdigit() or len(phone) != 10:
            flash("Invalid phone number. Must be a valid 10-digit number.")
            return redirect(url_for('auth.admin_dashboard'))

        # Handle photo upload
        photo_path = None
        filename = None
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename:
                filename = generate_secure_filename(photo.filename)
                photo_path = os.path.join('uploads', 'student_photos', filename)
                os.makedirs(os.path.dirname(photo_path), exist_ok=True)
                photo.save(photo_path)

                # Generate random password for student login
        password = generate_random_password()

        # Create user credentials entry
        credentials = UserCredentials(email=email, role="student")
        credentials.set_password(password)
        db.session.add(credentials)
        db.session.flush()  # Get credential ID before inserting student

        # Calculate current academic year (1st to 4th year)
        current_year = datetime.utcnow().year - int(admission_year) + 1
        current_year = max(1, min(current_year, 4))  # Ensuring it's between 1st to 4th year

        # Define batch-specific table name (Includes Course, Admission Year, Batch ID, and Semester)
        batch_table = f'student_batch_{course_id}_{admission_year}_{semester}_{batch_id}'

        # Insert student data into batch-semester-specific table
        sql = text(f"""
        INSERT INTO {batch_table} (
            credential_id, first_name, last_name, email, phone, address, photo_path,
            roll_number, current_year, admission_year, course_id, batch_id, semester, admission_date
        ) VALUES (
            :credential_id, :first_name, :last_name, :email, :phone, :address, :photo_path,
            :roll_number, :current_year, :admission_year, :course_id, :batch_id, :semester, :admission_date
        )
        """)

        db.session.execute(sql, {
            'credential_id': credentials.id,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone,
            'address': address,
            'photo_path': filename,
            'roll_number': roll_number,
            'current_year': current_year,
            'admission_year': int(admission_year),
            'course_id': course_id,
            'batch_id': batch_id,
            "semester": semester,
            'admission_date': datetime.utcnow()
        })

        db.session.commit()
        flash(f'Student added successfully. Temporary password: {password}')
        logging.info(f'New student created: {email}, Batch {batch_id}, Semester {semester}, Table: {batch_table}')

    except Exception as e:
        db.session.rollback()
        flash(f'Error adding student: {str(e)}')
        logging.error(f'Error creating student: {str(e)}')

    return redirect(url_for('auth.render_student'))


@bp.route('/admin/manage_batches')
@login_required
def manage_batches():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Fetch all existing student batch tables
    sql = text("""
    SELECT table_name FROM information_schema.tables
    WHERE table_name LIKE 'student_batch_%' AND table_schema = DATABASE()
    """)
    result = db.session.execute(sql)
    batch_tables = result.fetchall()

    student_batches = []
    for (table_name,) in batch_tables:
        parts = table_name.split('_')
        if len(parts) >= 4:
            course_id = parts[2]
            year = parts[3]
            semester = parts[4]
            id = parts[5]

            course = Course.query.get(course_id)
            if course:
                student_batches.append({
                    'table_name': table_name,
                    'course': course.name,
                    'year': year,
                    'id': id,
                    'semester': semester
                })

    return render_template('dashboard/manage_batches.html', student_batches=student_batches)


@bp.route('/admin/view_batch/<table_name>')
@login_required
def view_batch(table_name):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.manage_batches'))

    try:
        # Fetch table data dynamically
        sql = text(f"SELECT * FROM {table_name} ORDER BY CAST(roll_number AS UNSIGNED) ASC")
        result = db.session.execute(sql)
        rows = result.fetchall()

        # Get column names dynamically
        columns = [col[0] for col in db.session.execute(text(f"DESCRIBE {table_name}")).fetchall()]

        return render_template('dashboard/view_batch.html', table_name=table_name, rows=rows, columns=columns)

    except Exception as e:
        flash(f"Error retrieving batch data: {str(e)}")
        return redirect(url_for('auth.manage_batches'))


@bp.route('/admin/update_student/<table_name>/<int:student_id>', methods=['POST'])
@login_required
def update_student(table_name, student_id):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.manage_batches'))

    try:
        # Define allowed extensions and max file size (in bytes)

        UPLOAD_FOLDER = os.path.join('uploads', 'student_photos')

        # Get actual column names
        column_query = db.session.execute(text(f"DESCRIBE {table_name}"))
        columns = {col[0] for col in column_query.fetchall()}

        credential_query = db.session.execute(
            text(f"SELECT credential_id, email, photo_path FROM {table_name} WHERE id = :student_id"),
            {"student_id": student_id}
        )
        student_data = credential_query.fetchone()

        if student_data is None:
            flash('Error: Student record not found.')
            return redirect(url_for('auth.view_batch', table_name=table_name))

        credential_id, email, old_photo = student_data  # Get old photo path

        # Build update query for student table
        updates = []
        values = {"student_id": student_id}
        update_credentials = {}

        for key, value in request.form.items():
            if key in columns:
                updates.append(f"{key} = :{key}")
                values[key] = value

                if key == "email":
                    update_credentials[key] = value

        # 🖼️ **Handle Photo Upload with Validation**
        if "photo" in request.files:
            photo = request.files["photo"]
            if photo.filename:  # Check if file was uploaded
                filename_ext = photo.filename.rsplit(".", 1)[-1].lower()

                # Validate extension
                if filename_ext not in ALLOWED_EXTENSIONS:
                    flash("Invalid file type! Please upload a PNG, JPG, JPEG, or GIF image.")
                    return redirect(url_for('auth.view_batch', table_name=table_name))

                # Validate image type using imghdr
                photo.seek(0)  # Reset file pointer before checking
                file_type = imghdr.what(photo)
                if file_type not in ALLOWED_EXTENSIONS:
                    flash("Invalid image format! Ensure it's a valid image file.")
                    return redirect(url_for('auth.view_batch', table_name=table_name))

                # Validate file size
                photo.seek(0, os.SEEK_END)  # Move pointer to end to get file size
                file_size = photo.tell()
                if file_size > MAX_FILE_SIZE:
                    flash("File size exceeds 5MB limit! Please upload a smaller image.")
                    return redirect(url_for('auth.view_batch', table_name=table_name))

                photo.seek(0)  # Reset pointer for saving

                # Generate unique filename using email prefix
                if old_photo:
                    old_photo_path = os.path.join(UPLOAD_FOLDER, old_photo)
                    if os.path.exists(old_photo_path):
                        os.remove(old_photo_path)

                filename = generate_secure_filename(photo.filename)

                new_photo_path = os.path.join(UPLOAD_FOLDER, filename)
                photo.save(new_photo_path)  # Save the file

                # Delete old photo if it exists

                updates.append("photo_path = :photo_path")
                values["photo_path"] = filename

        if updates:
            sql = f"UPDATE {table_name} SET {', '.join(updates)} WHERE id = :student_id"
            db.session.execute(text(sql), values)

        # Update credentials if needed
        if update_credentials:
            credential_sql = "UPDATE user_credentials SET " + ", ".join(
                f"{key} = :{key}" for key in update_credentials) + " WHERE id = :id"
            update_credentials["id"] = credential_id
            db.session.execute(text(credential_sql), update_credentials)

        db.session.commit()
        flash("Student data updated successfully!")

    except Exception as e:
        db.session.rollback()
        flash(f"Error updating student data: {str(e)}")
        logging.error(f"Error updating student data: {str(e)}")

    return redirect(url_for('auth.view_batch', table_name=table_name))


@bp.route('/admin/delete_student/<table_name>/<int:student_id>', methods=['POST'])
@login_required
def delete_student(table_name, student_id):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.manage_batches'))

    try:
        # Retrieve the student's credential_id from the batch table before deleting
        student_query = db.session.execute(
            text(f"SELECT credential_id FROM {table_name} WHERE id = :student_id"),
            {"student_id": student_id}
        )
        student = student_query.fetchone()

        if student and student[0]:  # Ensure credential_id is found
            credential_id = student[0]

            # First, delete the student from the batch table
            db.session.execute(
                text(f"DELETE FROM {table_name} WHERE id = :student_id"),
                {"student_id": student_id}
            )

            # Check if the credential exists in `user_credentials`
            credential_check = db.session.execute(
                text("SELECT id FROM user_credentials WHERE id = :credential_id"),
                {"credential_id": credential_id}
            ).fetchone()

            if credential_check:
                # Delete credentials from `user_credentials`
                db.session.execute(
                    text("DELETE FROM user_credentials WHERE id = :credential_id"),
                    {"credential_id": credential_id}
                )
                flash('Student and credentials deleted successfully', 'success')
            else:
                flash('Student deleted, but credentials not found in user_credentials', 'warning')

            db.session.commit()

        else:
            flash('Student not found or credential ID missing', 'danger')

    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting student: {str(e)}', 'danger')
        logging.error(f'Error deleting student: {str(e)}')

    return redirect(url_for('auth.view_batch', table_name=table_name))


@bp.route('/admin/delete_student_batch/<batch_name>', methods=['POST'])
@login_required
def delete_student_batch(batch_name):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.manage_batches'))

    try:
        sql = text(f"DROP TABLE {batch_name}")
        db.session.execute(sql)
        db.session.commit()
        flash('Student batch deleted successfully')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting student batch: {str(e)}')

    return redirect(url_for('auth.manage_batches'))


@bp.route('/admin/manage_hods')
@login_required
def manage_hods():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.admin_dashboard'))

    # Fetch HOD details along with credentials and photo path
    sql = text("""
        SELECT h.id, h.credential_id, h.first_name, h.last_name, h.email, h.phone, h.address, 
               h.department, h.office_location, h.appointment_date, h.photo_path
        FROM hod_profiles h
        JOIN user_credentials u ON h.credential_id = u.id
    """)
    hods = db.session.execute(sql).fetchall()

    return render_template('dashboard/manage_hods.html', hods=hods)


@bp.route('/admin/manage_teachers')
@login_required
def manage_teachers():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required', 'danger')
        return redirect(url_for('auth.admin_dashboard'))

    try:
        # Fetch Teacher details along with credentials using raw SQL
        sql = text("""
            SELECT t.id, t.credential_id, t.first_name, t.last_name, t.phone, 
                   t.address, t.department, t.photo_path, u.email
            FROM teacher_details t
            JOIN user_credentials u ON t.credential_id = u.id
        """)
        teachers = db.session.execute(sql).fetchall()

        return render_template('dashboard/manage_teachers.html', teachers=teachers)

    except Exception as e:
        flash(f"Error fetching teacher details: {str(e)}", "danger")
        return redirect(url_for('auth.admin_dashboard'))


@bp.route('/admin/update_hod/<int:hod_id>', methods=['POST'])
@login_required
def update_hod(hod_id):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required', 'danger')
        return redirect(url_for('auth.manage_hods'))

    try:
        # Define upload folder
        UPLOAD_FOLDER = os.path.join('uploads', 'hod_photos')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Get existing HOD details
        hod_query = db.session.execute(
            text("SELECT credential_id, photo_path FROM hod_profiles WHERE id = :hod_id"),
            {"hod_id": hod_id}
        ).fetchone()

        if not hod_query:
            flash('HOD not found!', 'danger')
            return redirect(url_for('auth.manage_hods'))

        credential_id, old_photo_path = hod_query

        # Extract updated values from the form
        department = request.form.get("department")
        office_location = request.form.get("office_location")
        email = request.form.get("email")

        # Handle photo update
        new_photo = request.files.get("photo")
        photo_path = old_photo_path  # Default to old photo

        if new_photo and new_photo.filename:
            filename_ext = new_photo.filename.rsplit(".", 1)[-1].lower()

            # Validate file extension
            if filename_ext not in ALLOWED_EXTENSIONS:
                flash("Invalid file type! Please upload PNG, JPG, JPEG, or GIF.", "danger")
                return redirect(url_for('auth.manage_hods'))

            # Validate image format
            new_photo.seek(0)
            file_type = imghdr.what(new_photo)
            if file_type not in ALLOWED_EXTENSIONS:
                flash("Invalid image format! Ensure it's a valid image file.", "danger")
                return redirect(url_for('auth.manage_hods'))

            # Validate file size
            new_photo.seek(0, os.SEEK_END)
            file_size = new_photo.tell()
            if file_size > MAX_FILE_SIZE:
                flash("File size exceeds 5MB limit! Please upload a smaller image.", "danger")
                return redirect(url_for('auth.manage_hods'))
            new_photo.seek(0)

            # Secure filename and define path
            filename = generate_secure_filename(new_photo.filename)
            new_photo_path = os.path.join(UPLOAD_FOLDER, filename)

            # Delete old photo if it exists
            if old_photo_path:
                old_photo_full_path = os.path.join(UPLOAD_FOLDER, old_photo_path)
                if os.path.exists(old_photo_full_path):
                    os.remove(old_photo_full_path)

            # Save new photo
            new_photo.save(new_photo_path)
            photo_path = filename  # Update database with new filename

        # Update HOD details in database
        sql = """
        UPDATE hod_profiles 
        SET email=:email, department=:department, office_location=:office_location, photo_path=:photo_path
        WHERE id=:hod_id
        """
        db.session.execute(text(sql), {
            "email": email,
            "department": department,
            "office_location": office_location,
            "photo_path": photo_path,  # Updated photo path
            "hod_id": hod_id
        })

        # Update email in `user_credentials` table
        sql = """
        UPDATE user_credentials 
        SET email=:email 
        WHERE id=:credential_id
        """
        db.session.execute(text(sql), {"email": email, "credential_id": credential_id})

        db.session.commit()
        flash('HOD details and photo updated successfully!', 'success')

    except Exception as e:
        db.session.rollback()
        flash(f'Error updating HOD: {str(e)}', 'danger')

    return redirect(url_for('auth.manage_hods'))


@bp.route('/admin/delete_hod/<int:hod_id>', methods=['GET', 'POST'])
@login_required
def delete_hod(hod_id):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.manage_hods'))

    try:
        # Get the HOD's credential_id and photo path before deleting
        hod_query = db.session.execute(
            text("SELECT credential_id, photo_path FROM hod_profiles WHERE id = :hod_id"),
            {"hod_id": hod_id}
        )
        hod = hod_query.fetchone()  # Fetch only one row (assuming ID is unique)

        if hod:
            credential_id, photo_path = hod  # Extract values

            # Delete the profile photo if it exists
            if photo_path:
                photo_file_path = os.path.join(current_app.root_path, 'static', 'uploads', photo_path)
                if os.path.exists(photo_file_path):
                    os.remove(photo_file_path)

            # Delete the HOD record
            db.session.execute(text("DELETE FROM hod_profiles WHERE id = :hod_id"), {"hod_id": hod_id})

            # Delete from `user_credentials`
            db.session.execute(text("DELETE FROM user_credentials WHERE id = :credential_id"),
                               {"credential_id": credential_id})

            db.session.commit()
            flash('HOD, credentials, and profile photo deleted successfully', 'success')

        else:
            flash('HOD not found or credential ID missing', 'danger')

    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting HOD: {str(e)}', 'danger')

    return redirect(url_for('auth.manage_hods'))


# Allowed image extensions and max file size (5MB)


@bp.route('/admin/update_teacher/<int:teacher_id>', methods=['POST'])
@login_required
def update_teacher(teacher_id):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required', 'danger')
        return redirect(url_for('auth.manage_teachers'))

    try:
        # Fetch teacher credentials and photo path
        teacher_query = db.session.execute(
            text("SELECT credential_id, photo_path FROM teacher_details WHERE id = :teacher_id"),
            {"teacher_id": teacher_id}
        ).fetchone()

        if not teacher_query:
            flash('Teacher not found!', 'danger')
            return redirect(url_for('auth.manage_teachers'))

        credential_id, old_photo_path = teacher_query

        # Extract updated values from the form
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        address = request.form.get("address")
        department = request.form.get("department")

        # Define upload folder
        upload_folder = os.path.join('uploads', 'teacher_photos')
        os.makedirs(upload_folder, exist_ok=True)

        # Handle new photo upload
        new_photo = request.files.get("photo")
        photo_path = old_photo_path  # Default to existing photo

        if new_photo and new_photo.filename:
            filename_ext = new_photo.filename.rsplit(".", 1)[-1].lower()

            # Validate file extension
            if filename_ext not in ALLOWED_EXTENSIONS:
                flash("Invalid file type! Please upload PNG, JPG, JPEG, or GIF.", "danger")
                return redirect(url_for('auth.manage_teachers'))

            # Validate image format
            new_photo.seek(0)
            file_type = imghdr.what(new_photo)
            if file_type not in ALLOWED_EXTENSIONS:
                flash("Invalid image format! Ensure it's a valid image file.", "danger")
                return redirect(url_for('auth.manage_teachers'))

            # Validate file size
            new_photo.seek(0, os.SEEK_END)
            file_size = new_photo.tell()
            if file_size > MAX_FILE_SIZE:
                flash("File size exceeds 5MB limit! Please upload a smaller image.", "danger")
                return redirect(url_for('auth.manage_teachers'))
            new_photo.seek(0)

            # Secure filename
            filename = generate_secure_filename(new_photo.filename)
            new_photo_path = os.path.join(upload_folder, filename)

            # Remove old photo if exists
            if old_photo_path:
                old_photo_full_path = os.path.join(upload_folder, old_photo_path)
                if os.path.exists(old_photo_full_path):
                    os.remove(old_photo_full_path)

            # Save new photo
            new_photo.save(new_photo_path)
            photo_path = filename  # Update DB photo path

        # Update Teacher details
        sql = """
        UPDATE teacher_details 
        SET first_name=:first_name, last_name=:last_name, email=:email, 
            phone=:phone, address=:address, department=:department, photo_path=:photo_path
        WHERE id=:teacher_id
        """
        db.session.execute(text(sql), {
            "first_name": first_name, "last_name": last_name, "email": email,
            "phone": phone, "address": address, "department": department,
            "photo_path": photo_path,  # Updated photo path
            "teacher_id": teacher_id
        })

        # Update user_credentials table (email)
        sql = """
        UPDATE user_credentials 
        SET email=:email
        WHERE id=:credential_id
        """
        db.session.execute(text(sql), {
            "email": email,
            "credential_id": credential_id
        })

        db.session.commit()
        flash('Teacher updated successfully', 'success')

    except Exception as e:
        db.session.rollback()
        flash(f'Error updating Teacher: {str(e)}', 'danger')

    return redirect(url_for('auth.manage_teachers'))


@bp.route('/admin/delete_teacher/<int:teacher_id>', methods=['GET', 'POST'])
@login_required
def delete_teacher(teacher_id):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for('auth.manage_teachers'))

    try:
        # Get the teacher's credential_id and photo path before deleting
        teacher_query = db.session.execute(
            text("SELECT credential_id, photo_path FROM teacher_details WHERE id = :teacher_id"),
            {"teacher_id": teacher_id}
        )
        teacher = teacher_query.fetchone()

        if teacher:
            credential_id, photo_path = teacher  # Extract values from the query result

            # Delete the profile photo if it exists
            if photo_path:
                photo_file_path = os.path.join(current_app.root_path, 'static', 'uploads', 'teacher_photos', photo_path)
                if os.path.exists(photo_file_path):
                    os.remove(photo_file_path)

            # Delete the Teacher record
            db.session.execute(text("DELETE FROM teacher_details WHERE id = :teacher_id"), {"teacher_id": teacher_id})

            # Delete from `user_credentials`
            db.session.execute(text("DELETE FROM user_credentials WHERE id = :credential_id"),
                               {"credential_id": credential_id})

            db.session.commit()
            flash('Teacher, credentials, and profile photo deleted successfully', 'success')

        else:
            flash('Teacher not found', 'danger')

    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting Teacher: {str(e)}', 'danger')

    return redirect(url_for('auth.manage_teachers'))


@bp.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    courses = Course.query.all()
    course_subjects = CourseSubject.query.all()
    result = db.session.execute(text("SHOW TABLES LIKE 'student_batch_%'"))
    tables = [row[0] for row in result.fetchall()]  # Fetch all table names

    # Fetch all existing student batch data for dropdown handling

    return render_template(
        'dashboard/admin.html',
        courses=courses,
        course_subjects=course_subjects,
        table_names=tables
    )


@bp.route('/hod/dashboard')
@login_required
def hod_dashboard():
    courses = Course.query.all()
    teachers = TeacherDetails.query.all()
    subject_assignments = SubjectAssignment.query.all()
    return render_template('dashboard/hod_dashboard.html', courses=courses, teachers=teachers,
                           subject_assignments=subject_assignments)


@bp.route('/hod/manage_teacher_subjects')
@login_required
def manage_teacher_subjects():
    courses = Course.query.all()
    teachers = TeacherDetails.query.all()
    subject_assignments = SubjectAssignment.query.all()
    return render_template('dashboard/manage_teacher_subjects.html', courses=courses, teachers=teachers,
                           subject_assignments=subject_assignments)


@bp.route('/get_batches_years', methods=['GET'])
@login_required
def get_batches_years():
    course_id = request.args.get('course_id', type=int)
    if not course_id:
        return jsonify({'error': 'Missing course_id'}), 400

    years = db.session.query(CourseSubject.year).filter_by(course_id=course_id).distinct().all()
    return jsonify({'years': [y[0] for y in years]})


@bp.route('/get_semesters', methods=['GET'])
@login_required
def get_semesters():
    course_id = request.args.get('course_id', type=int)
    year = request.args.get('year', type=int)

    if not course_id or not year:
        return jsonify({'error': 'Missing parameters'}), 400

    semesters = db.session.query(CourseSubject.semester).filter_by(course_id=course_id, year=year).distinct().all()
    return jsonify({'semesters': [s[0] for s in semesters]})


@bp.route('/get_batches', methods=['GET'])
@login_required
def get_batches():
    course_id = request.args.get('course_id', type=int)
    year = request.args.get('year', type=int)
    semester = request.args.get('semester', type=int)

    if not course_id or not year or not semester:
        return jsonify({'error': 'Missing parameters'}), 400

    batches = db.session.query(CourseSubject.batch_id).filter_by(course_id=course_id, year=year,
                                                                 semester=semester).distinct().all()
    return jsonify({'batches': [b[0] for b in batches]})


@bp.route('/get_subjects_course', methods=['GET'])
@login_required
def get_subjects_course():
    course_id = request.args.get('course_id', type=int)
    year = request.args.get('year', type=int)
    semester = request.args.get('semester', type=int)
    batch_id = request.args.get('batch_id', type=int)

    if not course_id or not year or not semester or not batch_id:
        return jsonify({'error': 'Missing parameters'}), 400

    subjects = CourseSubject.query.filter_by(
        course_id=course_id, year=year, semester=semester, batch_id=batch_id, is_active=True
    ).all()

    return jsonify(
        {'subjects': [{'id': s.id, 'subject_code': s.subject_code, 'subject_name': s.subject_name} for s in subjects]})


@bp.route('/get_teachers', methods=['GET'])
@login_required
def get_teachers():
    teachers = TeacherDetails.query.all()
    return jsonify({'teachers': [{'id': t.id, 'name': f"{t.first_name} {t.last_name}"} for t in teachers]})


@bp.route('/assign_subject', methods=['POST'])
@login_required
def assign_subject():
    course_subject_id = request.form.get('subject_id', type=int)
    teacher_id = request.form.get('teacher_id', type=int)

    if not course_subject_id or not teacher_id:
        flash('Please select a subject and teacher.', 'danger')
        return redirect(url_for('auth.hod_dashboard'))

    existing_assignment = SubjectAssignment.query.filter_by(course_subject_id=course_subject_id,
                                                            teacher_id=teacher_id).first()
    if existing_assignment:
        flash('This subject is already assigned to the selected teacher.', 'warning')
        return redirect(url_for('auth.hod_dashboard'))

    assignment = SubjectAssignment(course_subject_id=course_subject_id, teacher_id=teacher_id)
    db.session.add(assignment)
    db.session.commit()

    flash('Subject assigned successfully!', 'success')
    return redirect(url_for('auth.hod_dashboard'))


@bp.route('/get_assignments', methods=['GET'])
@login_required
def get_assignments():
    assignments = SubjectAssignment.query.all()

    data = []
    for a in assignments:
        data.append({
            'course': a.course_subject.course.name,
            'subject_code': a.course_subject.subject_code,
            'subject_name': a.course_subject.subject_name,
            'year': a.course_subject.year,
            'semester': a.course_subject.semester,
            'batch_id': a.course_subject.batch_id,
            'teacher_name': f"{a.teacher.first_name} {a.teacher.last_name}",
        })

    return jsonify({'assignments': data})


@bp.route('/remove_subject_assignment', methods=['POST'])
@login_required
def remove_subject_assignment():
    assignment_id = request.form.get('assignment_id', type=int)
    print(assignment_id)

    if not assignment_id:
        flash('Invalid request.', 'danger')
        return redirect(url_for('auth.hod_dashboard'))

    assignment = SubjectAssignment.query.get(assignment_id)
    if not assignment:
        flash('Assignment not found.', 'warning')
        return redirect(url_for('auth.hod_dashboard'))

    db.session.delete(assignment)
    db.session.commit()

    flash('Assignment removed successfully.', 'success')
    return redirect(url_for('auth.hod_dashboard'))



@bp.route('/teacher/dashboard')
@login_required
def teacher_dashboard():
    if current_user.role != 'teacher':
        flash('Access denied: Teacher privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Get assigned subjects for the teacher along with all necessary details
    assigned_subjects = db.session.execute(text(
        """
        SELECT 
            cs.subject_name,
            cs.subject_code,
            c.name AS course_name,
            cs.year,
            cs.semester,
            cs.batch_id,
            cs.course_id
        FROM subject_assignments sa
        JOIN course_subjects cs ON sa.course_subject_id = cs.id
        JOIN courses c ON cs.course_id = c.id
        WHERE sa.teacher_id = :teacher_id
        """
    ), {"teacher_id": current_user.teacher_profile.id}).fetchall()
    for subject in assigned_subjects:
        print(subject)  # Prints each row as a tuple

    return render_template('dashboard/teacher.html', assigned_subjects=assigned_subjects)





@bp.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        flash('Access denied: Student privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Fetch session variables
    batch_id = session['batch_id']
    course_id = session['course_id']
    admission_year = session['admission_year']
    semester = session['semester']
    table_name = session['batch_table_name']
    email = current_user.email
    credential_id = current_user.id

    # Fetch student details
    query = text(
        f"SELECT id, first_name, last_name, roll_number FROM {table_name} WHERE credential_id = :credential_id")
    student = db.session.execute(query, {"credential_id": credential_id}).fetchone()

    if not student:
        flash("Student not found!", "error")
        return redirect(url_for('auth.logout'))

    student_id = student.id
    session['student_id']=student_id

    # Fetch subjects
    query = text("""
    SELECT subject_code, subject_name
    FROM course_subjects
    WHERE course_id = :course_id AND batch_id = :batch_id AND semester = :semester AND year = :admission_year
    """)
    subjects = db.session.execute(query, {
        "course_id": course_id,
        "batch_id": batch_id,
        "semester": semester,
        "admission_year": admission_year
    }).fetchall()

    # Fetch attendance

    attendance_table_name = f"attendance_{course_id}_{admission_year}"
    attendance_summary = {}

    for subject in subjects:
        query = text(f"""
            SELECT COUNT(*) FROM {attendance_table_name} 
        WHERE student_id = :student_id AND subject_code = :subject_code AND batch_id = :batch_id AND semester = :semester and year=:admission_year
        """)
        total_classes = db.session.execute(query, {
            "student_id": student_id,
            "subject_code": subject.subject_code,
            "batch_id": batch_id,
            "semester": semester,
            "admission_year": admission_year
        }).scalar() or 1  # Avoid division by zero

        query = text(f"""
            SELECT COUNT(*) FROM {attendance_table_name} 
        WHERE student_id = :student_id AND subject_code = :subject_code AND batch_id = :batch_id AND semester = :semester and status='Present'
        """)
        present_count = db.session.execute(query, {
            "student_id": student_id,
            "subject_code": subject.subject_code,
            "batch_id": batch_id,
            "semester": semester
        }).scalar() or 0

        attendance_percentage = (present_count / total_classes) * 100

        attendance_summary[subject.subject_code] = {
            "present_count": present_count,
            "total_classes": total_classes,
            "percentage": round(attendance_percentage, 2),
        }

    # Fetch teacher details
    teacher_data = {}

    for subject in subjects:
        query = text("""
        SELECT t.first_name, t.last_name, t.email 
        FROM subject_assignments sa
        JOIN teacher_details t ON sa.teacher_id = t.id
        JOIN course_subjects cs ON sa.course_subject_id = cs.id
        WHERE cs.subject_code = :subject_code AND cs.batch_id = :batch_id 
              AND cs.semester = :semester AND cs.year = :admission_year AND cs.course_id = :course_id
        """)
        teacher = db.session.execute(query, {
            "subject_code": subject.subject_code,
            "batch_id": batch_id,
            "semester": semester,
            "admission_year": admission_year,
            "course_id": course_id
        }).fetchone()
        teacher_data[subject.subject_code] = teacher

    return render_template("dashboard/student.html", student=student, subjects=subjects,
                           attendance_summary=attendance_summary, teacher_data=teacher_data,admission_year=session.get("admission_year"),
                           semester=session.get("semester"),
                           batch_id=session.get("batch_id"))




@bp.route("/download_attendance/<subject_code>")
@login_required
def download_attendance(subject_code):
    attendance_table_name = f"attendance_{session['course_id']}_{session['admission_year']}"
    student_id = session.get('student_id')
    batch_id = session.get('batch_id')
    admission_year = session.get('admission_year')
    semester = session.get('semester')

    query = text(f"""
        SELECT date, status FROM {attendance_table_name}
        WHERE student_id = :student_id AND subject_code = :subject_code 
              AND batch_id = :batch_id AND year = :admission_year 
              AND semester = :semester
    """)
    records = db.session.execute(query, {
        "student_id": student_id,
        "subject_code": subject_code,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester
    }).fetchall()

    # Create CSV response
    def generate_csv():
        yield "Date,Status\n"
        for record in records:
            status =  record.status
            yield f"{record.date},{status}\n"

    return Response(generate_csv(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{subject_code}.csv"})




@bp.route('/student/view_notes/<subject_code>', methods=['GET'])
@login_required
def view_notes(subject_code):
    # Security Check: Ensure session data matches
    if not all(key in session for key in ['batch_id', 'course_id', 'admission_year', 'semester', 'student_id', 'batch_table_name']):
        flash("Session data is missing or corrupted!", "danger")
        return redirect(url_for('auth.student_dashboard'))

    batch_id = session['batch_id']
    course_id = session['course_id']
    admission_year = session['admission_year']
    semester = session['semester']
    student_id = session['student_id']
    table_name = session['batch_table_name']

    # Validate subject exists for this batch, semester, and course
    result = db.session.execute(text("""
        SELECT id,subject_name FROM course_subjects
        WHERE course_id = :course_id 
        AND batch_id = :batch_id
        AND year = :admission_year
        AND semester = :semester
        AND subject_code = :subject_code
        AND is_active = 1
    """), {
        "course_id": course_id,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester,
        "subject_code": subject_code
    }).fetchone()

    if not result:
        flash("Unauthorized access: Subject not found or not assigned to this batch!", "danger")
        return redirect(url_for('auth.student_dashboard'))

    # Subject is valid → Get Notes
    BASE_UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    subject_folder = os.path.join(BASE_UPLOAD_FOLDER, f"{course_id}/{admission_year}/semester_{semester}/batch_{batch_id}/subject_{subject_code}")

    existing_notes = os.listdir(subject_folder) if os.path.exists(subject_folder) else []
    print(existing_notes)
    print("hi")

    return render_template('dashboard/view_notes.html',
                           batch_id=batch_id,
                           admission_year=admission_year,
                           semester=semester,
                           course_id=course_id,
                           subject_code=subject_code,
                           subject_name=result.subject_name,
                           existing_notes=existing_notes)

@bp.route('/student/download_note_student/<subject_code>/<filename>', methods=['GET'])
@login_required
def download_note_student(subject_code, filename):
    # Security Check: Ensure session data matches
    if not all(key in session for key in ['batch_id', 'course_id', 'admission_year', 'semester', 'student_id', 'batch_table_name']):
        flash("Session data is missing or corrupted!", "danger")
        return redirect(url_for('auth.student_dashboard'))

    batch_id = session['batch_id']
    course_id = session['course_id']
    admission_year = session['admission_year']
    semester = session['semester']
    student_id = session['student_id']
    table_name = session['batch_table_name']

    # Validate subject exists for this batch, semester, and course
    result = db.session.execute(text("""
        SELECT id FROM course_subjects
        WHERE course_id = :course_id 
        AND batch_id = :batch_id
        AND year = :admission_year
        AND semester = :semester
        AND subject_code = :subject_code
        AND is_active = 1
    """), {
        "course_id": course_id,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester,
        "subject_code": subject_code
    }).fetchone()

    if not result:
        flash("Unauthorized access: Subject not found or not assigned to this batch!", "danger")
        return redirect(url_for('auth.student_dashboard'))

    # Subject is valid → Allow download
    BASE_UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    subject_folder = os.path.join(BASE_UPLOAD_FOLDER, f"{course_id}/{admission_year}/semester_{semester}/batch_{batch_id}/subject_{subject_code}")
    file_path = os.path.join(subject_folder, filename)

    if not os.path.exists(file_path):
        flash("File not found!", "danger")
        return redirect(url_for('student.view_notes', subject_code=subject_code))

    return send_file(file_path, as_attachment=True)










@bp.route('/admin/add_course', methods=['POST'])
@login_required
def add_course():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    try:
        course = Course(
            code=request.form.get('course_code'),
            name=request.form.get('course_name')
        )
        db.session.add(course)
        db.session.commit()
        flash('Course added successfully')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding course: {str(e)}')
        logging.error(f'Error adding course: {str(e)}')

    return redirect(url_for('auth.admin_dashboard'))


@bp.route('/admin/add_course_subject', methods=['POST'])
@login_required
def add_course_subject():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Capture form data
    print(request.form)
    course_id = request.form.get('course_id')
    year = request.form.get('year')
    semester = request.form.get('semester')
    batch_id = request.form.get('batch_id')
    subject_code = request.form.get('subject_code')
    subject_name = request.form.get('subject_name')

    # Debugging: Print received form data
    print(
        f"Received data: course_id={course_id}, year={year}, semester={semester}, batch_id={batch_id}, subject_code={subject_code}, subject_name={subject_name}")

    # Validate required fields
    if not all([course_id, year, semester, batch_id, subject_code, subject_name]):
        flash('All fields are required!')
        return redirect(url_for('auth.admin_dashboard'))

    try:
        # Ensure year is converted to an integer
        year = int(year)

        # Check if subject already exists
        existing_subject = CourseSubject.query.filter_by(
            course_id=course_id, year=year, semester=semester, batch_id=batch_id, subject_code=subject_code
        ).first()

        if existing_subject:
            flash('Subject already exists for the selected course and batch!')
        else:
            new_subject = CourseSubject(
                course_id=course_id,
                year=year,  # Ensuring it's an integer
                semester=int(semester),
                batch_id=batch_id,
                subject_code=subject_code,
                subject_name=subject_name
            )

            db.session.add(new_subject)
            db.session.commit()
            flash('Subject added successfully')
            print(f"Successfully added subject: {new_subject}")

    except Exception as e:
        db.session.rollback()
        flash(f'Error adding subject: {str(e)}')
        logging.error(f'Error adding subject: {str(e)}')

    return redirect(url_for('auth.admin_dashboard'))


@bp.route('/admin/database')
@login_required
def database_management():
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Get all models from SQLAlchemy
    models_list = []
    for cls in db.Model._decl_class_registry.values():
        if isinstance(cls, type) and issubclass(cls, db.Model):
            if hasattr(cls, '__tablename__'):
                count = db.session.query(cls).count()
                models_list.append({
                    'name': cls.__name__,
                    'table': cls.__tablename__,
                    'columns': [column.name for column in cls.__table__.columns],
                    'count': count
                })

    return render_template('dashboard/database.html', models=models_list)


@bp.route('/admin/database/<table_name>')
@login_required
def view_table(table_name):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Get the model class for the table
    model = None
    for cls in db.Model._decl_class_registry.values():
        if isinstance(cls, type) and issubclass(cls, db.Model):
            if hasattr(cls, '__tablename__') and cls.__tablename__ == table_name:
                model = cls
                break

    if not model:
        flash(f'Table {table_name} not found')
        return redirect(url_for('auth.database_management'))

    # Get search parameter
    search = request.args.get('search', '').strip()

    # Get table data with optional search
    query = model.query
    if search:
        # Create search conditions for string columns
        conditions = []
        for column in model.__table__.columns:
            if isinstance(column.type, db.String):
                conditions.append(column.ilike(f'%{search}%'))
        if conditions:
            query = query.filter(or_(*conditions))

    records = query.all()
    columns = [column.name for column in model.__table__.columns]

    return render_template('dashboard/table_view.html',
                           table_name=table_name,
                           columns=columns,
                           records=records)


@bp.route('/admin/database/<table_name>/schema')
@login_required
def view_schema(table_name):
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    # Get the model class for the table
    model = None
    for cls in db.Model._decl_class_registry.values():
        if isinstance(cls, type) and issubclass(cls, db.Model):
            if hasattr(cls, '__tablename__') and cls.__tablename__ == table_name:
                model = cls
                break

    if not model:
        return jsonify({'error': 'Table not found'}), 404

    # Get schema information
    inspector = inspect(db.engine)
    columns = []
    for column in inspector.get_columns(table_name):
        columns.append({
            'name': column['name'],
            'type': str(column['type']),
            'nullable': column['nullable'],
            'default': str(column['default']) if column['default'] else None,
            'primary_key': column['primary_key']
        })

    return jsonify({'columns': columns})


@bp.route('/admin/database/<table_name>/export')
@login_required
def export_table(table_name):
    if current_user.role != 'admin':
        flash('Access denied: Admin privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Get the model class for the table
    model = None
    for cls in db.Model._decl_class_registry.values():
        if isinstance(cls, type) and issubclass(cls, db.Model):
            if hasattr(cls, '__tablename__') and cls.__tablename__ == table_name:
                model = cls
                break

    if not model:
        flash(f'Table {table_name} not found')
        return redirect(url_for('auth.database_management'))

    # Create CSV data
    si = StringIO()
    writer = csv.writer(si)

    # Write headers
    columns = [column.name for column in model.__table__.columns]
    writer.writerow(columns)

    # Write data
    records = model.query.all()
    for record in records:
        row = [getattr(record, column) for column in columns]
        writer.writerow(row)

    output = si.getvalue()
    si.close()

    # Create the response
    return send_file(
        StringIO(output),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{table_name}.csv'
    )


def create_attendance_table(course_id, admission_year):
    table_name = f"attendance_{course_id}_{admission_year}"
    db.session.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT NOT NULL,
            subject_code VARCHAR(50) NOT NULL,
            batch_id INT NOT NULL,
            semester INT NOT NULL,
            year INT NOT NULL,
            date DATE NOT NULL,
            status ENUM('Present', 'Absent') NOT NULL,
            teacher_id INT NOT NULL,
            UNIQUE(student_id, subject_code, batch_id, semester, year, date)
        )
    """))
    db.session.commit()


@bp.route('/teacher/take_attendance', methods=['GET', 'POST'])
@login_required
def take_attendance():
    if current_user.role != 'teacher':
        flash('Access denied: Teacher privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Get subject details from URL parameters
    subject_code = request.args.get('subject_code')
    batch_id = request.args.get('batch_id')
    admission_year = request.args.get('admission_year')
    course_id = request.args.get("course_id")
    semester = request.args.get('semester')
    year = admission_year  # New field added

    if not all([subject_code, batch_id, admission_year, semester, course_id, year]):
        flash('Invalid request parameters')
        return redirect(url_for('auth.teacher_dashboard'))

    # Ensure attendance table exists
    create_attendance_table(course_id, admission_year)

    # Fetch students from batch table
    student_table = f"student_batch_{course_id}_{admission_year}_{semester}_{batch_id}"
    students = db.session.execute(text(f"SELECT id, first_name, last_name, roll_number FROM {student_table} ORDER BY roll_number")).fetchall()
    students_list = [dict(row._mapping) for row in students]

    if request.method == 'POST':
        date = request.form['date']
        teacher_id = current_user.teacher_profile.id
        attendance_records = []
        existing_attendance_count = 0  # Track if attendance already exists

        # Check if attendance exists for this batch, semester, year, and subject
        table_name = f"attendance_{course_id}_{admission_year}"
        existing_records = db.session.execute(text(f"""
            SELECT COUNT(*) FROM {table_name} 
            WHERE batch_id = :batch_id AND subject_code = :subject_code 
            AND semester = :semester AND year = :year AND date = :date
        """), {"batch_id": batch_id, "subject_code": subject_code, "semester": semester, "year": year, "date": date}).scalar()

        if existing_records > 0:
            existing_attendance_count = existing_records

        for student in students:
            status = request.form.get(f"attendance_{student.id}", "Absent")
            attendance_records.append({
                "student_id": student.id,
                "subject_code": subject_code,
                "batch_id": batch_id,
                "semester": semester,
                "year": year,
                "date": date,
                "status": status,
                "teacher_id": teacher_id
            })

        # Insert or update attendance records
        for record in attendance_records:
            db.session.execute(text(f"""
                INSERT INTO {table_name} (student_id, subject_code, batch_id, semester, year, date, status, teacher_id)
                VALUES (:student_id, :subject_code, :batch_id, :semester, :year, :date, :status, :teacher_id)
                ON DUPLICATE KEY UPDATE status = :status, teacher_id = :teacher_id
            """), record)

        db.session.commit()

        # Flash the appropriate message
        if existing_attendance_count > 0:
            flash('Student attendance updated successfully!', 'success')
        else:
            flash('Student attendance inserted successfully!', 'success')

        return redirect(url_for('auth.teacher_dashboard'))

    return render_template('dashboard/take_attendance.html', students=students_list, subject_code=subject_code, batch_id=batch_id, admission_year=admission_year, semester=semester, course_id=course_id, year=year)



@bp.route('/teacher/get_attendance', methods=['GET'])
@login_required
def get_attendance():
    if current_user.role != 'teacher':
        return jsonify({"error": "Unauthorized access"}), 403

    # Extract parameters from request
    date = request.args.get('date')
    subject_code = request.args.get('subject_code')
    batch_id = request.args.get('batch_id')
    admission_year = request.args.get('admission_year')
    semester = request.args.get('semester')
    course_id = request.args.get("course_id")
    year = request.args.get("year")

    print(f"Received GET request: date={date}, subject_code={subject_code}, batch_id={batch_id}, admission_year={admission_year}, semester={semester}, course_id={course_id}, year={year}")

    if not all([date, subject_code, batch_id, admission_year, semester, course_id, year]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Define the attendance table based on course and admission year
    table_name = f"attendance_{course_id}_{admission_year}"

    # Query attendance records for the given date and subject
    query = text(f"""
        SELECT student_id, status FROM {table_name}
        WHERE date = :date AND subject_code = :subject_code
        AND batch_id = :batch_id AND semester = :semester AND year = :year
    """)

    attendance_records = db.session.execute(query, {
        "date": date,
        "subject_code": subject_code,
        "batch_id": batch_id,
        "semester": semester,
        "year": year
    }).fetchall()

    print("Fetched attendance records:", attendance_records)

    # Convert query results into a list of dictionaries
    attendance_data = [{"student_id": row.student_id, "status": row.status} for row in attendance_records]

    return jsonify(attendance_data)





@bp.route('/teacher/attendance_report', methods=['GET'])
@login_required
def attendance_report():
    if current_user.role != 'teacher':
        flash('Access denied: Teacher privileges required')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    # Get required filters from query parameters
    subject_code = request.args.get('subject_code')
    batch_id = request.args.get('batch_id')
    admission_year = request.args.get('admission_year')
    course_id = request.args.get('course_id')
    semester = request.args.get('semester')
    start_date_str = request.args.get('start_date')  # e.g., 2024-02-01
    end_date_str = request.args.get('end_date')  # e.g., 2024-02-10

    if not all([subject_code, batch_id, admission_year, semester, course_id, start_date_str, end_date_str]):
        flash('Invalid request parameters')
        return redirect(url_for('auth.teacher_dashboard'))

    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # Generate all dates between start_date and end_date
    date_range = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in
                  range((end_date - start_date).days + 1)]

    # Define the dynamic table name
    attendance_table = f"attendance_{course_id}_{admission_year}"
    student_table = f"student_batch_{course_id}_{admission_year}_{semester}_{batch_id}"

    # Fetch student details
    student_query = text(f"""
        SELECT s.roll_number, s.first_name, s.last_name
        FROM {student_table} s
        ORDER BY s.roll_number
    """)

    students = db.session.execute(student_query).fetchall()

    # Create dictionary to store attendance with "A" as default
    attendance_data = {s.roll_number: {"Name": f"{s.first_name} {s.last_name}"} for s in students}
    print(attendance_data)
    # Initialize all date columns with "A" (Absent)
    for student in attendance_data:
        for date in date_range:
            attendance_data[student][date] = "A"

    # Fetch attendance records
    attendance_query = text(f"""
        SELECT s.roll_number, a.date, a.status
        FROM {attendance_table} a
        JOIN {student_table} s ON a.student_id = s.id
        WHERE a.batch_id = :batch_id 
        AND a.subject_code = :subject_code 
        AND a.semester = :semester 
        AND a.date BETWEEN :start_date AND :end_date
    """)

    attendance_records = db.session.execute(attendance_query, {
        "batch_id": batch_id,
        "subject_code": subject_code,
        "semester": semester,
        "start_date": start_date_str,
        "end_date": end_date_str
    }).fetchall()

    # Update the attendance dictionary with "P" where present
    for roll_number, date, status in attendance_records:
        print(roll_number,date,status)
        attendance_data[roll_number][date.strftime("%Y-%m-%d")] = "P" if status == "Present" else "A"

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(attendance_data, orient="index", columns=["Name"] + date_range)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Roll Number"}, inplace=True)  # Rename index column

    # Generate Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Attendance Report", index=False)

    output.seek(0)

    # Send the Excel file as response
    filename = f"Attendance_Report_{batch_id}_{start_date_str}_to_{end_date_str}.xlsx"
    return send_file(output, download_name=filename, as_attachment=True,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")







@bp.route('/teacher/manage_notes/<int:batch_id>/<int:admission_year>/<int:semester>/<int:course_id>/<subject_code>', methods=['GET', 'POST'])
@login_required
def manage_notes(batch_id, admission_year, semester, course_id, subject_code):
    if current_user.role != 'teacher':
        flash('Access denied!', 'danger')
        return redirect(url_for(f'auth.{current_user.role}_dashboard'))

    teacher_id = current_user.teacher_profile.id

    # Security Check: Ensure the teacher is assigned to this subject
    result = db.session.execute(text("""
        SELECT cs.id 
        FROM subject_assignments sa
        JOIN course_subjects cs ON sa.course_subject_id = cs.id
        WHERE sa.teacher_id = :teacher_id 
          AND cs.batch_id = :batch_id 
          AND cs.year = :admission_year
          AND cs.semester = :semester
          AND cs.course_id = :course_id
          AND cs.subject_code = :subject_code
    """), {
        "teacher_id": teacher_id,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester,
        "course_id": course_id,
        "subject_code": subject_code
    }).fetchone()

    if not result:
        flash('Unauthorized access: You are not assigned to this subject!', 'danger')
        return redirect(url_for('auth.teacher_dashboard'))

    BASE_UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    # Define subject folder path
    subject_folder = os.path.join(BASE_UPLOAD_FOLDER, f"{course_id}/{admission_year}/semester_{semester}/batch_{batch_id}/subject_{subject_code}")

    # Create directory if not exists
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    # List existing notes
    existing_notes = os.listdir(subject_folder)

    return render_template('dashboard/manage_notes.html',
                           batch_id=batch_id,
                           admission_year=admission_year,
                           semester=semester,
                           course_id=course_id,
                           subject_code=subject_code,
                           existing_notes=existing_notes)


upload_progress = {}


@bp.route('/teacher/upload_progress/<upload_id>', methods=['GET'])
@login_required
def get_upload_progress(upload_id):
    def generate():
        # Initial progress message
        yield f"data: {json.dumps({'progress': 0, 'status': 'Starting upload'})}\n\n"

        # Keep checking progress until complete
        while True:
            if upload_id in upload_progress:
                progress_data = upload_progress[upload_id]
                yield f"data: {json.dumps(progress_data)}\n\n"

                # If processing is complete, break the loop
                if progress_data.get('status') == 'complete' or progress_data.get('status') == 'error':
                    # Clean up the progress data
                    if upload_id in upload_progress:
                        del upload_progress[upload_id]
                    break

            time.sleep(0.5)  # Check every 500ms

    # Return a streaming response
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@bp.route('/teacher/upload_notes', methods=['POST'])
@login_required
def upload_notes():
    # Create a unique ID for this upload session
    import uuid
    upload_id = str(uuid.uuid4())

    # Initialize progress for this upload
    upload_progress[upload_id] = {
        'progress': 0,
        'status': 'Starting',
        'processed_files': 0,
        'total_files': 0
    }

    if current_user.role != 'teacher':
        upload_progress[upload_id] = {'progress': 0, 'status': 'error', 'message': 'Access denied!'}
        return jsonify({'success': False, 'upload_id': upload_id, 'message': 'Access denied!'})

    ALLOWED_EXTENSIONS = {
        'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp', 'pdf', 'txt',
        'rtf', 'tex', 'csv', 'tsv', 'epub', 'mobi', 'jpg', 'jpeg', 'png', 'gif', 'bmp',
        'tiff', 'svg', 'eps', 'webp', 'heic', 'raw', 'psd', 'ai', 'py', 'java', 'c',
        'cpp', 'cs', 'go', 'rs', 'rb', 'swift', 'sh', 'bat', 'cmd', 'pl', 'r', 'html',
        'css', 'js', 'ts', 'php', 'asp', 'jsp', 'json', 'yaml', 'yml', 'xml', 'ini',
        'cfg', 'toml', 'sql', 'db', 'sqlite', 'md', 'rst', 'adoc', 'gitignore',
        'gitattributes', 'lua', 'kt', 'dart', 'm', 'asm'
    }

    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB limit per file

    # Get form data
    batch_id = request.form.get('batch_id')
    admission_year = request.form.get('admission_year')
    semester = request.form.get('semester')
    course_id = request.form.get('course_id')
    subject_code = request.form.get('subject_code')
    files = request.files.getlist('files')  # Get multiple files

    # Update progress - received request
    upload_progress[upload_id]['status'] = 'Validating request'
    upload_progress[upload_id]['progress'] = 5

    if not all([batch_id, admission_year, semester, course_id, subject_code]) or not files:
        upload_progress[upload_id] = {'progress': 0, 'status': 'error', 'message': 'All fields are required!'}
        return jsonify({'success': False, 'upload_id': upload_id, 'message': 'All fields are required!'})

    # Update total files count
    upload_progress[upload_id]['total_files'] = len(files)

    # Update progress - checking teacher assignment
    upload_progress[upload_id]['status'] = 'Checking permissions'
    upload_progress[upload_id]['progress'] = 10

    # Ensure teacher is assigned to this subject
    teacher_id = current_user.teacher_profile.id
    result = db.session.execute(text("""
        SELECT cs.id 
        FROM subject_assignments sa
        JOIN course_subjects cs ON sa.course_subject_id = cs.id
        WHERE sa.teacher_id = :teacher_id 
          AND cs.batch_id = :batch_id 
          AND cs.year = :admission_year
          AND cs.semester = :semester
          AND cs.course_id = :course_id
          AND cs.subject_code = :subject_code
    """), {
        "teacher_id": teacher_id,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester,
        "course_id": course_id,
        "subject_code": subject_code
    }).fetchone()

    if not result:
        upload_progress[upload_id] = {'progress': 0, 'status': 'error', 'message': 'Unauthorized access!'}
        return jsonify({'success': False, 'upload_id': upload_id, 'message': 'Unauthorized access!'})

    # Update progress - preparing directory
    upload_progress[upload_id]['status'] = 'Preparing file storage'
    upload_progress[upload_id]['progress'] = 15

    # Define folder path
    BASE_UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    subject_folder = os.path.join(BASE_UPLOAD_FOLDER,
                                  f"{course_id}/{admission_year}/semester_{semester}/batch_{batch_id}/subject_{subject_code}")

    os.makedirs(subject_folder, exist_ok=True)

    # Calculate progress increment per file
    progress_per_file = 75 / max(len(files), 1)  # 15% to 90% for file processing

    uploaded_files = []
    for index, file in enumerate(files, start=1):
        # Update progress for each file
        file_progress = 15 + (index * progress_per_file)
        filename = file.filename if file.filename else "unknown"
        upload_progress[upload_id] = {
            'progress': int(file_progress),
            'status': f'Processing file {index + 1} of {len(files)}: {filename}',
            'processed_files': index,
            'total_files': len(files)
        }

        if file.filename == "":
            continue

        # Security: Check file type
        original_filename = secure_filename(file.filename)

        # Split the filename into name and extension
        name, ext = os.path.splitext(original_filename)

        # Add a timestamp to make it unique
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}{ext}"
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if file_extension not in ALLOWED_EXTENSIONS:
            continue

        # Security: Check file size
        if file.content_length > MAX_FILE_SIZE:
            continue

        # Save file securely
        upload_progress[upload_id]['status'] = f'Saving file: {filename}'
        file_path = os.path.join(subject_folder, filename)
        file.save(file_path)
        uploaded_files.append(filename)

        # Update progress after saving
        upload_progress[upload_id] = {
            'progress': int(file_progress + (progress_per_file / 2)),
            'status': f'Saved file {index + 1} of {len(files)}: {filename}',
            'processed_files': index + 1,
            'total_files': len(files)
        }

    # Final progress update
    if uploaded_files:
        upload_progress[upload_id] = {
            'progress': 100,
            'status': 'complete',
            'message': f'Uploaded: {", ".join(uploaded_files)}',
            'processed_files': len(uploaded_files),
            'total_files': len(files)
        }
        flash(f'Uploaded: {", ".join(uploaded_files)}', 'success')
    else:
        upload_progress[upload_id] = {
            'progress': 100,
            'status': 'error',
            'message': 'No valid files were uploaded.',
            'processed_files': 0,
            'total_files': len(files)
        }
        flash('No valid files were uploaded.', 'danger')

    # Return a JSON response with the upload_id
    return jsonify({
        'success': True,
        'upload_id': upload_id,
        'message': 'Upload complete',
        'files': uploaded_files
    })



@bp.route('/teacher/download_file/<int:course_id>/<int:admission_year>/<int:semester>/<int:batch_id>/<subject_code>/<filename>')
@login_required
def download_file(course_id, admission_year, semester, batch_id, subject_code, filename):
    teacher_id = current_user.teacher_profile.id

    # Security Check
    result = db.session.execute(text("""
        SELECT cs.id 
        FROM subject_assignments sa
        JOIN course_subjects cs ON sa.course_subject_id = cs.id
        WHERE sa.teacher_id = :teacher_id 
          AND cs.batch_id = :batch_id 
          AND cs.year = :admission_year
          AND cs.semester = :semester
          AND cs.course_id = :course_id
          AND cs.subject_code = :subject_code
    """), {
        "teacher_id": teacher_id,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester,
        "course_id": course_id,
        "subject_code": subject_code
    }).fetchone()

    if not result:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('teacher.teacher_dashboard'))

    BASE_UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

    subject_folder = os.path.join(BASE_UPLOAD_FOLDER, f"{course_id}/{admission_year}/semester_{semester}/batch_{batch_id}/subject_{subject_code}")

    return send_from_directory(subject_folder, filename, as_attachment=True)


@bp.route('/teacher/delete_note', methods=['POST'])
@login_required
def delete_note():
    teacher_id = current_user.teacher_profile.id

    # Fetching values from the form
    file_name = request.form.get("file_name")
    course_id = request.form.get("course_id")
    admission_year = request.form.get("admission_year")
    semester = request.form.get("semester")
    batch_id = request.form.get("batch_id")
    subject_code = request.form.get("subject_code")

    # Convert numeric values to integers (important to prevent type issues)
    course_id = int(course_id)
    admission_year = int(admission_year)
    semester = int(semester)
    batch_id = int(batch_id)

    # Security Check: Verify if the teacher is assigned to this subject
    result = db.session.execute(text("""
        SELECT cs.id 
        FROM subject_assignments sa
        JOIN course_subjects cs ON sa.course_subject_id = cs.id
        WHERE sa.teacher_id = :teacher_id 
          AND cs.batch_id = :batch_id 
          AND cs.year = :admission_year
          AND cs.semester = :semester
          AND cs.course_id = :course_id
          AND cs.subject_code = :subject_code
    """), {
        "teacher_id": teacher_id,
        "batch_id": batch_id,
        "admission_year": admission_year,
        "semester": semester,
        "course_id": course_id,
        "subject_code": subject_code
    }).fetchone()

    if not result:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('teacher.teacher_dashboard'))

    # File deletion process
    BASE_UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

    subject_folder = os.path.join(BASE_UPLOAD_FOLDER, f"{course_id}/{admission_year}/semester_{semester}/batch_{batch_id}/subject_{subject_code}")
    file_path = os.path.join(subject_folder, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)
        flash('File deleted successfully!', 'success')
    else:
        flash('File not found!', 'danger')

    return redirect(request.referrer)












@bp.route('/api/courses', methods=['GET'])
def get_courses():
    # Get all unique courses
    courses = db.session.query(
        BatchTable.course_id,
        BatchTable.course_name,
        BatchTable.course_code
    ).distinct().all()

    result = []
    for course in courses:
        result.append({
            'id': course.course_id,
            'course_name': course.course_name,
            'course_code': course.course_code
        })

    return jsonify(result)


@bp.route('/api/years', methods=['GET'])
def get_years():
    course_id = request.args.get('course_id')

    if not course_id:
        return jsonify([])

    # Get all admission years for the selected course
    years = db.session.query(
        BatchTable.admission_year
    ).filter(
        BatchTable.course_id == course_id
    ).distinct().order_by(
        BatchTable.admission_year.desc()
    ).all()

    return jsonify([year[0] for year in years])


@bp.route('/api/semesters', methods=['GET'])
def get_semesters_inlogin():
    course_id = request.args.get('course_id')
    admission_year = request.args.get('admission_year')

    if not course_id or not admission_year:
        return jsonify([])

    # Get all semesters for the selected course and admission year
    semesters = db.session.query(
        BatchTable.semester
    ).filter(
        BatchTable.course_id == course_id,
        BatchTable.admission_year == admission_year
    ).distinct().order_by(
        BatchTable.semester
    ).all()

    return jsonify([semester[0] for semester in semesters])


@bp.route('/api/batches', methods=['GET'])
def get_batches_inlogin():
    course_id = request.args.get('course_id')
    admission_year = request.args.get('admission_year')
    semester = request.args.get('semester')

    if not course_id or not admission_year or not semester:
        return jsonify([])

    # Get all batches for the selected course, admission year, and semester
    batches = db.session.query(
        BatchTable.id,
        BatchTable.batch_id
    ).filter(
        BatchTable.course_id == course_id,
        BatchTable.admission_year == admission_year,
        BatchTable.semester == semester
    ).distinct().all()

    result = []
    for batch in batches:
        result.append({
            'id': batch.id,
            'batch_id': batch.batch_id
        })

    return jsonify(result)




# ADMIN DASTABASE MANAGEMENT

@bp.route('/admin/database-explorer')
@login_required
def database_explorer():
    # Ensure only admin can access
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('auth.index'))

    # Get all table names from the database
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()

    return render_template('dashboard/database_explorer.html', tables=tables)


@bp.route('/api/table/<table_name>', methods=['GET'])
@login_required
def get_table_data(table_name):
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # Get table structure
        inspector = inspect(db.engine)
        columns = []
        for column in inspector.get_columns(table_name):
            is_primary = False
            for pk in inspector.get_pk_constraint(table_name)['constrained_columns']:
                if column['name'] == pk:
                    is_primary = True
                    break

            columns.append({
                "name": column['name'],
                "type": str(column['type']),
                "primary_key": is_primary
            })

        # Get table data
        query = f"SELECT * FROM {table_name} LIMIT 1000"  # Add limit for safety
        result = db.session.execute(text(query))
        rows = [dict(row._mapping) for row in result]

        return jsonify({"columns": columns, "rows": rows})

    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/table/<table_name>/structure', methods=['GET'])
@login_required
def get_table_structure(table_name):
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # Get table structure
        inspector = inspect(db.engine)
        columns = []
        for column in inspector.get_columns(table_name):
            is_primary = False
            is_auto_increment = False

            # Check if this is a primary key
            for pk in inspector.get_pk_constraint(table_name)['constrained_columns']:
                if column['name'] == pk:
                    is_primary = True
                    break

            # Attempt to determine auto increment (this can be database-specific)
            if is_primary and hasattr(column.get('type', None), 'autoincrement'):
                is_auto_increment = True

            columns.append({
                "name": column['name'],
                "type": str(column['type']),
                "primary_key": is_primary,
                "auto_increment": is_auto_increment
            })

        return jsonify(columns)

    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/table/<table_name>/row/<row_id>', methods=['PUT'])
@login_required
def update_row(table_name, row_id):
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.json

        # Get primary key column
        inspector = inspect(db.engine)
        pk_columns = inspector.get_pk_constraint(table_name)['constrained_columns']

        if not pk_columns:
            return jsonify({"success": False, "message": "Table has no primary key"}), 400

        # Build update query
        set_clause = ", ".join([f"{key} = :{key}" for key in data.keys()])
        where_clause = f"{pk_columns[0]} = :id"

        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

        # Add the ID to the parameters
        params = {**data, "id": row_id}

        # Execute the query
        result = db.session.execute(text(query), params)
        db.session.commit()

        return jsonify({"success": True, "message": f"Row updated successfully"})

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route('/api/table/<table_name>/row/<row_id>', methods=['DELETE'])
@login_required
def delete_row(table_name, row_id):
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # Get primary key column
        inspector = inspect(db.engine)
        pk_columns = inspector.get_pk_constraint(table_name)['constrained_columns']

        if not pk_columns:
            return jsonify({"success": False, "message": "Table has no primary key"}), 400

        # Build delete query
        query = f"DELETE FROM {table_name} WHERE {pk_columns[0]} = :id"

        # Execute the query
        result = db.session.execute(text(query), {"id": row_id})
        db.session.commit()

        return jsonify({"success": True, "message": "Row deleted successfully"})

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route('/api/table/<table_name>/row', methods=['POST'])
@login_required
def add_row(table_name):
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.json

        # Build insert query
        columns = ", ".join(data.keys())
        values = ", ".join([f":{key}" for key in data.keys()])

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

        # Execute the query
        result = db.session.execute(text(query), data)
        db.session.commit()

        return jsonify({"success": True, "message": "Row added successfully"})

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route('/api/execute-sql', methods=['POST'])
@login_required
def execute_sql():
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        query = request.json.get('query', '').strip()

        if not query:
            return jsonify({"success": False, "message": "No query provided"}), 400

        # Block potentially dangerous operations (this is a basic protection)
        dangerous_operations = ['DROP DATABASE', 'DROP SCHEMA', 'TRUNCATE DATABASE']
        for op in dangerous_operations:
            if op.upper() in query.upper():
                return jsonify({
                    "success": False,
                    "message": f"Operation not allowed: {op}"
                }), 403

        # Execute the query
        result = db.session.execute(text(query))

        # Try to determine which tables might be affected
        affected_tables = []
        query_upper = query.upper()
        if any(op in query_upper for op in ['UPDATE', 'DELETE', 'INSERT', 'ALTER']):
            # Extract table names from query (simplified approach)
            words = query.split()
            for i, word in enumerate(words):
                if word.upper() in ['INTO', 'UPDATE', 'FROM', 'TABLE'] and i + 1 < len(words):
                    table = words[i + 1].strip('`;,')
                    if table not in affected_tables:
                        affected_tables.append(table)

        # Check if the query returns data
        if result.returns_rows:
            # Get column names
            columns = [{"name": col} for col in result.keys()]

            # Get rows (limit to 1000 for safety)
            rows = [dict(row._mapping) for row in result.fetchmany(1000)]

            return jsonify({
                "success": True,
                "columns": columns,
                "rows": rows,
                "affectedTables": affected_tables
            })
        else:
            # For non-SELECT queries
            rowcount = result.rowcount
            db.session.commit()

            return jsonify({
                "success": True,
                "message": f"Query executed successfully. Rows affected: {rowcount}",
                "affectedTables": affected_tables
            })

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route('/api/table/<table_name>/export-csv')
@login_required
def export_csv(table_name):
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # Get table data
        query = f"SELECT * FROM {table_name}"
        result = db.session.execute(text(query))

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(result.keys())

        # Write data
        for row in result:
            writer.writerow(row)

        # Prepare response
        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename={table_name}.csv"}
        )

    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500



# MANAGE SUBJECTS
    # ✅ Route to render the Manage Subjects page (View Only)
@bp.route('/view_subjects')
@login_required
def view_subjects_page():
    if current_user.role != 'admin':  # Security Check: Only Admin Can Access
        flash("Access denied! Only admins can manage subjects.", "danger")
        return redirect(url_for('dashboard'))  # Redirect to a safe page

    subjects = CourseSubject.query.all()  # Fetch all subjects
    return render_template('dashboard/manage_subjects.html', subjects=subjects)

# ✅ Route to update a subject
@bp.route('/update_subject/<int:id>', methods=['POST'])
@login_required
def update_subject_details(id):
    if current_user.role != 'admin':  # Security Check
        flash("Access denied! Only admins can update subjects.", "danger")
        return redirect(url_for('auth.view_subjects_page'))

    subject = CourseSubject.query.get_or_404(id)

    subject.subject_code = request.form.get('subject_code')
    subject.subject_name = request.form.get('subject_name')
    subject.course_id = request.form.get('course_id')
    subject.batch_id = request.form.get('batch_id')
    subject.semester = request.form.get('semester')
    subject.year = request.form.get('year')

    db.session.commit()
    flash("Subject updated successfully!", "success")
    return redirect(url_for('auth.view_subjects_page'))

# ✅ Route to delete a subject
@bp.route('/delete_subject/<int:id>', methods=['POST'])
@login_required
def delete_subject_entry(id):
    if current_user.role != 'admin':  # Security Check
        flash("Access denied! Only admins can delete subjects.", "danger")
        return redirect(url_for('auth.view_subjects_page'))

    subject = CourseSubject.query.get_or_404(id)
    db.session.delete(subject)
    db.session.commit()
    flash("Subject deleted successfully!", "danger")
    return redirect(url_for('auth.view_subjects_page'))


# View Courses Page
@bp.route('/manage_courses')
@login_required
def manage_courses():
    courses = Course.query.all()
    return render_template('dashboard/manage_courses.html', courses=courses)


# Update Course
@bp.route('/update_course/<int:id>', methods=['POST'])
@login_required
def update_course(id):
    data = request.json
    course = Course.query.get(id)

    if course:
        course.code = data.get('code')
        course.name = data.get('name')
        course.is_active = bool(int(data.get('is_active')))
        db.session.commit()
        return jsonify({"message": "Course updated successfully"}), 200

    return jsonify({"message": "Course not found"}), 404


# Delete Course
@bp.route('/delete_course/<int:id>', methods=['POST'])
@login_required
def delete_course(id):
    course = Course.query.get(id)

    if course:
        db.session.delete(course)
        db.session.commit()
        return jsonify({"message": "Course deleted successfully"}), 200

    return jsonify({"message": "Course not found"}), 404


@bp.route('/forgot_password', methods=['POST'])
def forgot_password():
    email = request.form.get('email')
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Find the user by email
    user = UserCredentials.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Generate a new random password
    new_password = generate_random_password()

    # Hash the new password
    user.set_password(new_password)

    # Update the database
    try:
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'error': 'Database error'}), 500

    # Send the new password via email
    try:
        send_email(email, new_password)
    except Exception as e:
        return jsonify({'error': 'Failed to send email'}), 500

    return jsonify({'message': 'A new password has been sent to your email'}), 200


def send_email(to_email, new_password):
    # Email configuration
    smtp_server = 'smtp.example.com'
    smtp_port = 587
    smtp_user = 'your_email@example.com'
    smtp_password = 'your_email_password'

    # Create the email content
    msg = MIMEText(f'Your new password is: {new_password}')
    msg['Subject'] = 'Your New Password'
    msg['From'] = smtp_user
    msg['To'] = to_email

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())