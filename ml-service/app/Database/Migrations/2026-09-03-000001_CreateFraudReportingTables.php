<?php

namespace App\Database\Migrations;

use CodeIgniter\Database\Migration;

class CreateFraudReportingTables extends Migration
{
    public function up()
    {
        // 1. Tabel Users
        $this->forge->addField([
            'id'            => ['type' => 'INT', 'constraint' => 11, 'unsigned' => true, 'auto_increment' => true],
            'name'          => ['type' => 'VARCHAR', 'constraint' => 100],
            'email'         => ['type' => 'VARCHAR', 'constraint' => 100],
            'password_hash' => ['type' => 'VARCHAR', 'constraint' => 255],
            'role'          => ['type' => 'ENUM', 'constraint' => ['admin', 'investigator'], 'default' => 'investigator'],
            'created_at'    => ['type' => 'DATETIME', 'null' => true],
            'updated_at'    => ['type' => 'DATETIME', 'null' => true],
        ]);
        $this->forge->addKey('id', true);
        $this->forge->addUniqueKey('email');
        $this->forge->createTable('users', true);

        // 2. Tabel Transactions
        $this->forge->addField([
            'id'             => ['type' => 'BIGINT', 'constraint' => 20, 'unsigned' => true, 'auto_increment' => true],
            'transaction_id' => ['type' => 'BIGINT', 'constraint' => 20],
            'transaction_dt' => ['type' => 'INT', 'constraint' => 11, 'null' => true],
            'transaction_amt'=> ['type' => 'DECIMAL', 'constraint' => '15,2', 'default' => 0.00],
            'product_cd'     => ['type' => 'VARCHAR', 'constraint' => 20, 'null' => true],
            'addr1'          => ['type' => 'VARCHAR', 'constraint' => 50, 'null' => true],
            'addr2'          => ['type' => 'VARCHAR', 'constraint' => 50, 'null' => true],
            'device_type'    => ['type' => 'VARCHAR', 'constraint' => 50, 'null' => true],
            'device_info'    => ['type' => 'VARCHAR', 'constraint' => 100, 'null' => true],
            'raw_features'   => ['type' => 'JSON', 'null' => true],
            'is_fraud_label' => ['type' => 'TINYINT', 'constraint' => 1, 'null' => true],
            'data_split'     => ['type' => 'ENUM', 'constraint' => ['train', 'test'], 'null' => true],
            'created_at'     => ['type' => 'DATETIME', 'null' => true],
        ]);
        $this->forge->addKey('id', true);
        $this->forge->addKey('transaction_id');
        $this->forge->createTable('transactions', true);

        // 3. Tabel Detection Results
        $this->forge->addField([
            'id'             => ['type' => 'BIGINT', 'constraint' => 20, 'unsigned' => true, 'auto_increment' => true],
            'transaction_id' => ['type' => 'BIGINT', 'constraint' => 20],
            'anomaly_score'  => ['type' => 'FLOAT'],
            'status'         => ['type' => 'ENUM', 'constraint' => ['normal', 'anomali']],
            'model_version'  => ['type' => 'VARCHAR', 'constraint' => 100],
            'created_at'     => ['type' => 'DATETIME', 'null' => true],
        ]);
        $this->forge->addKey('id', true);
        $this->forge->addKey('transaction_id');
        $this->forge->createTable('detection_results', true);

        // 4. Tabel Narrative Reports
        $this->forge->addField([
            'id'                  => ['type' => 'BIGINT', 'constraint' => 20, 'unsigned' => true, 'auto_increment' => true],
            'detection_result_id' => ['type' => 'BIGINT', 'constraint' => 20, 'unsigned' => true],
            'narrative_text'      => ['type' => 'TEXT'],
            'triggered_rules'     => ['type' => 'JSON'],
            'created_at'          => ['type' => 'DATETIME', 'null' => true],
        ]);
        $this->forge->addKey('id', true);
        $this->forge->addForeignKey('detection_result_id', 'detection_results', 'id', 'CASCADE', 'CASCADE');
        $this->forge->createTable('narrative_reports', true);
    }

    public function down()
    {
        $this->forge->dropTable('narrative_reports', true);
        $this->forge->dropTable('detection_results', true);
        $this->forge->dropTable('transactions', true);
        $this->forge->dropTable('users', true);
    }
} 