using MachineLearning;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni per l'accesso ai gestori di dati
   /// </summary>
   public static class DataViewExtensions
   {
      #region Methods
      /// <summary>
      /// Crea una IDataView che e' il merge dell'oggetto principale con una serie di altre IDataView
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <param name="others">Altre viste di dati da concatenare</param>
      /// <returns>La vista di dati concatenata</returns>
      public static IDataView Merge(this IDataView dataView, IMachineLearningContextProvider context, params IDataView[] others) =>
         DataViewMerged.Create(context, dataView.Schema, new[] { dataView }.Concat(others).ToArray());
      /// <summary>
      /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
      /// returns true if current and other vector types have the same size and item type.
      /// </summary>
      public static bool SameSizeAndItemType(this DataViewType columnType, DataViewType other)
      {
         if (other == null)
            return false;
         if (columnType.Equals(other))
            return true;
         // For vector types, we don't care about the factoring of the dimensions.
         if (columnType is not VectorDataViewType vectorType || other is not VectorDataViewType otherVectorType)
            return false;
         if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
            return false;
         return vectorType.Size == otherVectorType.Size;
      }
      /// <summary>
      /// Trasforma la IDataView in una data view filtrata
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <param name="filter">Filtro</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewFiltered ToDataViewFiltered(this IDataView dataView, IMachineLearningContextProvider context, DataViewRowFilter filter) => DataViewFiltered.Create(context, dataView, filter);
      /// <summary>
      /// Trasforma la IDataView in una griglia di dati
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewGrid ToDataViewGrid(this IDataView dataView, IMachineLearningContextProvider context) => DataViewGrid.Create(context, dataView);
      /// <summary>
      /// Trasforma un cursore in enumerabile
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <returns>L'enumerabile</returns>
      public static IEnumerable<DataViewRow> ToEnumerable(this IDataView dataView)
      {
         var cursor = dataView.GetRowCursor(dataView.Schema);
         while (cursor.MoveNext())
            yield return cursor;
      }
      #endregion
   }
}
