﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reflection;

namespace MachineLearning.Data
{
   /// <summary>
   /// Griglia di dati
   /// </summary>
   public partial class DataViewGrid : IDataAccess, IEnumerable<DataViewValuesRow>
   {
      #region Fields
      /// <summary>
      /// Method info per l'ottenimento del getter
      /// </summary>
      private static readonly MethodInfo _getterMethodInfo = typeof(DataViewGrid).GetMethod(nameof(GetValue), BindingFlags.NonPublic | BindingFlags.Static);
      /// <summary>
      /// Lista di righe
      /// </summary>
      private readonly List<DataViewValuesRow> _rows;
      #endregion
      #region Properties
      /// <summary>
      /// Abilitazione allo shuffle
      /// </summary>
      public bool CanShuffle => false;
      /// <summary>
      /// Colonne della tabella
      /// </summary>
      public ReadOnlyCollection<Col> Cols { get; private set; }
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MachineLearningContext ML { get; private set; }
      /// <summary>
      /// Righe della tabella
      /// </summary>
      public ReadOnlyCollection<DataViewValuesRow> Rows => _rows.AsReadOnly();
      /// <summary>
      /// Schema dei dati
      /// </summary>
      public DataViewSchema Schema { get; private set; }
      /// <summary>
      /// Indicizzatore di colonne
      /// </summary>
      /// <param name="col">Colonna</param>
      /// <returns>La colonna</returns>
      public Col this[DataViewSchema.Column col] => Cols[col.Index];
      /// <summary>
      /// Indicizzatore di righe
      /// </summary>
      /// <param name="rowIndex"></param>
      /// <returns>La riga</returns>
      public DataViewValuesRow this[int rowIndex] => _rows[rowIndex];
      /// <summary>
      /// Indicizzatore di colonne
      /// </summary>
      /// <param name="columnName">Nome colonna</param>
      /// <returns>La colonna</returns>
      public Col this[string columnName] => this[Schema[columnName]];
      #endregion
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="schema">Lo schema della vista di dati</param>
      /// <param name="data">Vista di dati</param>
      private DataViewGrid(IMachineLearningContextProvider context, DataViewSchema schema, IDataAccess data)
      {
         // Check
         MachineLearningContext.AssertMLNET(context, nameof(context));
         ML = context.ML;
         ML.NET.Assert(schema != null || data != null, $"The parameter {nameof(schema)} or the parameter {nameof(data)} must be specified");
         if (schema != null && data != null) {
            ML.NET.Assert(
               schema.Zip(data.Schema).All(item => item.First.Type.SameSizeAndItemType(item.Second.Type)),
               $"The {nameof(schema)} and the {nameof(data)}.{nameof(data.Schema)} are different");
         }
         // Memorizza lo schema
         Schema = data?.Schema ?? schema;
         // Numero di colonne
         var n = Schema.Count;
         // Crea la lista di righe
         _rows = new List<DataViewValuesRow>();
         if (data != null) {
            // Creatore dei getter di valori riga
            var getter = new Func<DataViewRowCursor, int, object>[n];
            for (var i = 0; i < n; i++) {
               var getterGenericMethodInfo = _getterMethodInfo.MakeGenericMethod(Schema[i].Type.RawType);
               getter[i] = new Func<DataViewRowCursor, int, object>((cursor, col) => getterGenericMethodInfo.Invoke(null, new object[] { cursor, col }));
            }
            // Ottiene il cursore per la data view di input e itera su tutte le righe
            var cursor = data.GetRowCursor(Schema);
            while (cursor.MoveNext()) {
               // Valori
               var objects = new object[n];
               var active = new bool[n];
               // Legge la riga
               for (var i = 0; i < n; i++) {
                  objects[i] = getter[i](cursor, i);
                  active[i] = cursor.IsColumnActive(Schema[i]);
               }
               // Aggiunge la riga di dati
               var id = default(DataViewRowId);
               if (cursor.GetIdGetter() is var idGetter && idGetter != null)
                  idGetter(ref id);
               _rows.Add(DataViewValuesRow.Create(context, cursor.Schema, cursor.Position, id, objects, active));
            }
         }
         // Memorizza righe
         Cols = Array.AsReadOnly((from col in Schema select new Col(this, col.Index)).ToArray());
      }
      /// <summary>
      /// Aggiunge una riga alla griglia di dati
      /// </summary>
      /// <param name="values">Valori</param>
      public void Add(params object[] values)
      {
         ML.NET.AssertNonEmpty(values);
         ML.NET.Assert(values.Length == Schema.Count, $"The length of {nameof(values)} must be equal to the length of schema");
         for (var i = 0; i < values.Length; i++) {
            ML.NET.Assert(DataViewValue.CanConvert(values[i].GetType(), Schema[i].Type.RawType), $"Expected {Schema[i].Type.RawType} convertible value, got {values[i].GetType()} in column {Schema[i].Name}");
            values[i] = DataViewValue.Convert(values[i], Schema[i].Type.RawType);
         }
         _rows.Add(DataViewValuesRow.Create(this, Schema, _rows.Count, default, values, Enumerable.Range(0, values.Length).Select(i => true).ToArray()));
      }
      /// <summary>
      /// Aggiunge una riga alla griglia di dati
      /// </summary>
      /// <param name="values">Valori</param>
      public void Add(params DataViewValue[] values) => Add(values.Select(v => v.Value).ToArray());
      /// <summary>
      /// Aggiunge una riga alla griglia di dati
      /// </summary>
      /// <param name="values">Valori</param>
      public void Add(params (string Name, object Value)[] values)
      {
         ML.NET.CheckNonEmpty(values, nameof(values));
         var orderedValues = new object[Schema.Count];
         for (var i = 0; i < orderedValues.Length; i++) {
            ML.NET.CheckNonEmpty(values[i].Name, $"{nameof(values)}[{i}]", "The name cannot be null");
            var col = Schema.FirstOrDefault(c => c.Name == values[i].Name);
            ML.NET.CheckNonEmpty(col.Name, $"{nameof(values)}[{i}]", $"The schema doesn't contain the column {values[i].Name}");
            orderedValues[i] = values[i].Value;
         }
         Add(orderedValues);
      }
      /// <summary>
      /// Crea una griglia di dati a partire da una vista di dati
      /// </summary>
      /// <param name="data">Vista di dati</param>
      /// <returns>La griglia di dati</returns>
      public static DataViewGrid Create(IDataAccess data)
      {
         MachineLearningContext.CheckMLNET(data, nameof(data));
         data.ML.NET.CheckValue(data, nameof(data));
         return new DataViewGrid(data, null, data);
      }
      /// <summary>
      /// Crea una griglia di dati a partire da uno schema
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="schema">Schema</param>
      /// <returns>La griglia di dati</returns>
      public static DataViewGrid Create(IMachineLearningContextProvider context, DataViewSchema schema)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         context.ML.NET.CheckValue(schema, nameof(schema));
         return new DataViewGrid(context, schema, null);
      }
      /// <summary>
      /// Enumeratore di righe
      /// </summary>
      /// <returns>L'enumeratore</returns>
      public IEnumerator<DataViewValuesRow> GetEnumerator() => _rows.GetEnumerator();
      /// <summary>
      /// Restituisce il numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() => _rows.Count;
      /// <summary>
      /// Legge un valore dal cursore della dataview
      /// </summary>
      /// <typeparam name="T">Tipo di valore</typeparam>
      /// <param name="cursor">Cursore</param>
      /// <param name="col">Colonna</param>
      /// <returns>Il valore</returns>
      private static T GetValue<T>(DataViewRowCursor cursor, int col)
      {
         // Azione di restituzione dei valori
         var getter = cursor.GetGetter<T>(cursor.Schema[col]);
         T value = default;
         getter(ref value);
         return value;
      }
      /// <summary>
      /// Restituisce un cursore
      /// </summary>
      /// <param name="columnsNeeded">Colonne richieste</param>
      /// <param name="rand">Randomizzatore</param>
      /// <returns>Il cursore di linea</returns>
      public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand) => new Cursor(this);
      /// <summary>
      /// Restituisce un set di cursori
      /// </summary>
      /// <param name="columnsNeeded">Colonne richieste</param>
      /// <param name="rand">Randomizzatore</param>
      /// <returns>Il cursore di linea</returns>
      public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand) => new[] { (this as IDataAccess).GetRowCursor(columnsNeeded, rand) };
      /// <summary>
      /// Enumeratore di righe
      /// </summary>
      /// <returns>L'enumeratore</returns>
      IEnumerator IEnumerable.GetEnumerator() => _rows.GetEnumerator();
   }

   public partial class DataViewGrid // Cursor
   {
      /// <summary>
      /// Implementazione del cursore della data view
      /// </summary>
      private class Cursor : DataViewRowCursor
      {
         #region Fields
         /// <summary>
         /// Posizione
         /// </summary>
         private readonly DataViewGrid _owner;
         /// <summary>
         /// Posizione
         /// </summary>
         private long _position = -1;
         #endregion
         #region Properties
         /// <summary>
         /// Posizione
         /// </summary>
         public override long Position => _position;
         /// <summary>
         /// Indicatore batch
         /// </summary>
         public override long Batch => 0;
         /// <summary>
         /// Schema della vista di dati
         /// </summary>
         public override DataViewSchema Schema => _owner.Schema;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         internal Cursor(DataViewGrid owner) => _owner = owner;
         /// <summary>
         /// Funzione di restituzione del getter di valori
         /// </summary>
         /// <typeparam name="TValue">Tipo di valore</typeparam>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Il getter</returns>
         public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
         {
            if (!typeof(TValue).IsAssignableFrom(column.Type.RawType))
               throw _owner.ML.NET.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}', expected type: '{column.Type.RawType}'.");
            return (ref TValue value) => value = (TValue)_owner.Rows[(int)Position].Values[column.Index].Value;
         }
         /// <summary>
         /// Restituzione del getter dell'identificativo
         /// </summary>
         /// <returns>Il getter dell'identificativo</returns>
         public override ValueGetter<DataViewRowId> GetIdGetter() => (ref DataViewRowId id) => id = _owner.Rows[(int)Position].Id;
         /// <summary>
         /// Indica se la colonna e' attiva
         /// </summary>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Stato di attivita'</returns>
         public override bool IsColumnActive(DataViewSchema.Column column) => _owner._rows[(int)Position].IsColumnActive(column);
         /// <summary>
         /// Muove il cursore alla posizione successiva
         /// </summary>
         /// <returns>true se posizione successiva esistente</returns>
         public override bool MoveNext()
         {
            if (_position < _owner.GetRowCount() - 1) {
               _position++;
               return true;
            }
            return false;
         }
         #endregion
      }
   }

   public partial class DataViewGrid // Col
   {
      /// <summary>
      /// Colonna
      /// </summary>
      public class Col : IEnumerable<DataViewValue>
      {
         #region Fields
         /// <summary>
         /// Indice di colonna
         /// </summary>
         private readonly int _index;
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly DataViewGrid _owner;
         #endregion
         #region Properties
         /// <summary>
         /// Indicizzatore
         /// </summary>
         /// <param name="row">Indice di riga</param>
         /// <returns>Il valore</returns>
         public DataViewValue this[int row] => _owner._rows[row][_index];
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         internal Col(DataViewGrid owner, int index)
         {
            _owner = owner;
            _index = index;
         }
         /// <summary>
         /// Enumeratore di valori
         /// </summary>
         /// <returns>L'enumeratore</returns>
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
         /// <summary>
         /// Enumeratore di valori
         /// </summary>
         /// <returns>L'enumeratore</returns>
         public IEnumerator<DataViewValue> GetEnumerator() => (from row in _owner._rows select row[_index]).GetEnumerator();
         #endregion
      }
   }

}
